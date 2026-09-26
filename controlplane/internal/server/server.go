// Package server implements the ControlPlane gRPC service.
package server

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
	"github.com/r7bb/Model-Passport/controlplane/internal/audit"
	"github.com/r7bb/Model-Passport/controlplane/internal/auth"
	"github.com/r7bb/Model-Passport/controlplane/internal/store"
)

// Lifecycle states in which each environment may serve a version (see the Python lifecycle).
var (
	devStates      = []string{"clean", "canary", "verifying"}
	consumerStates = []string{"approved", "released"}
	killableStates = []string{"findings", "clean", "canary", "verifying", "approved", "released"}
)

// Caller is an authenticated user acting within one organization.
type Caller struct {
	UserID     string
	Email      string
	SuperAdmin bool
	Tenant     store.Tenant
	Role       string
}

func (c Caller) actor() audit.Actor {
	id := c.UserID
	return audit.Actor{ID: &id, Label: c.Email}
}

type callerKey struct{}

// Server is the ControlPlane implementation.
type Server struct {
	pb.UnimplementedControlPlaneServer
	Store  *store.Store
	Secret string
}

func first(md metadata.MD, key string) string {
	if values := md.Get(key); len(values) > 0 {
		return values[0]
	}
	return ""
}

// Authenticate is a unary interceptor: it verifies the session, resolves the organization
// from "x-mp-tenant", and checks membership before any handler runs.
func (s *Server) Authenticate(ctx context.Context, req any, info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler) (any, error) {
	if strings.HasPrefix(info.FullMethod, "/grpc.health.v1.Health/") {
		return handler(ctx, req) // load balancers probe health without a session
	}
	md, _ := metadata.FromIncomingContext(ctx)
	claims, err := auth.Verify(first(md, "authorization"), s.Secret)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}
	slug := first(md, "x-mp-tenant")
	if slug == "" {
		return nil, status.Error(codes.InvalidArgument, "missing x-mp-tenant")
	}
	email, err := s.Store.UserEmail(ctx, claims.Subject)
	if err != nil {
		return nil, status.Error(codes.Unauthenticated, "account not found or disabled")
	}
	tenant, role, err := s.Store.Member(ctx, slug, claims.Subject)
	switch {
	case errors.Is(err, store.ErrNotFound):
		return nil, status.Errorf(codes.NotFound, "organization %q not found", slug)
	case err != nil:
		return nil, status.Error(codes.Internal, err.Error())
	case role == "" && !claims.SuperAdmin:
		return nil, status.Error(codes.PermissionDenied, "you are not a member of this organization")
	case tenant.Status == "suspended" && !claims.SuperAdmin:
		return nil, status.Error(codes.PermissionDenied, "this organization is suspended")
	}
	caller := Caller{UserID: claims.Subject, Email: email, SuperAdmin: claims.SuperAdmin,
		Tenant: tenant, Role: role}
	return handler(context.WithValue(ctx, callerKey{}, caller), req)
}

func caller(ctx context.Context, permission string) (Caller, error) {
	c, ok := ctx.Value(callerKey{}).(Caller)
	if !ok {
		return c, status.Error(codes.Unauthenticated, "not authenticated")
	}
	if !auth.Allowed(c.Role, permission, c.SuperAdmin) {
		return c, status.Errorf(codes.PermissionDenied, "your role cannot do this (needs %s)", permission)
	}
	return c, nil
}

func environment(env pb.Environment) (string, string, error) {
	switch env {
	case pb.Environment_DEV:
		return "dev", auth.DevEndpoints, nil
	case pb.Environment_CONSUMER:
		return "consumer", auth.ReleasesManage, nil
	default:
		return "", "", status.Error(codes.InvalidArgument, "environment must be DEV or CONSUMER")
	}
}

func toProto(d store.Deployment) *pb.Deployment {
	env := pb.Environment_DEV
	if d.Environment == "consumer" {
		env = pb.Environment_CONSUMER
	}
	return &pb.Deployment{Id: d.ID, VersionId: d.VersionID, ModelId: d.ModelID, Model: d.Model,
		Version: d.Version, Environment: env, Status: d.Status, Endpoint: d.Endpoint,
		CreatedAt: d.CreatedAt.UTC().Format("2006-01-02T15:04:05Z")}
}

func grpcError(err error) error {
	if errors.Is(err, store.ErrNotFound) {
		return status.Error(codes.NotFound, "not found in this organization")
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	return status.Error(codes.Internal, err.Error())
}

// Deploy serves a version on developer endpoints or to consumers, replacing what served before.
func (s *Server) Deploy(ctx context.Context, req *pb.DeployRequest) (*pb.Deployment, error) {
	env, permission, err := environment(req.GetEnvironment())
	if err != nil {
		return nil, err
	}
	c, err := caller(ctx, permission)
	if err != nil {
		return nil, err
	}
	var created store.Deployment
	err = s.Store.InTenant(ctx, c.Tenant.ID, func(tx pgx.Tx) error {
		v, err := store.GetVersion(ctx, tx, c.Tenant.ID, req.GetVersionId())
		if err != nil {
			return err
		}
		allowed := devStates
		if env == "consumer" {
			allowed = consumerStates
		}
		if v.State == "killed" {
			return status.Error(codes.FailedPrecondition, "this version is blocked by the kill switch")
		}
		if !slices.Contains(allowed, v.State) {
			return status.Errorf(codes.FailedPrecondition,
				"a %s version cannot be deployed to %s (needs one of %v)", v.State, env, allowed)
		}
		var replaced []string
		rows, err := tx.Query(ctx, `UPDATE deployments d SET status = 'superseded'
			FROM model_versions v WHERE v.id = d.version_id AND d.tenant_id = $1
			AND v.model_id = $2 AND d.environment = $3 AND d.status = 'active' RETURNING d.id`,
			c.Tenant.ID, v.ModelID, env)
		if err != nil {
			return err
		}
		if replaced, err = pgx.CollectRows(rows, pgx.RowTo[string]); err != nil {
			return err
		}
		created = store.Deployment{ID: uuid.NewString(), VersionID: v.ID, ModelID: v.ModelID,
			Model: v.Model, Version: v.Version, Environment: env, Status: "active",
			Endpoint: fmt.Sprintf("/v1/%s/%s", env, v.Model)}
		err = tx.QueryRow(ctx, `INSERT INTO deployments (id, tenant_id, version_id, environment,
			status, endpoint, created_by, created_at) VALUES ($1, $2, $3, $4, 'active', $5, $6, now())
			RETURNING created_at`, created.ID, c.Tenant.ID, v.ID, env, created.Endpoint, c.UserID).
			Scan(&created.CreatedAt)
		if err != nil {
			return err
		}
		replacedAny := make([]any, len(replaced))
		for i, id := range replaced {
			replacedAny[i] = id
		}
		return audit.Record(ctx, tx, c.Tenant.ID, c.actor(), "deployment.created", "deployment",
			created.ID, map[string]any{"environment": env, "model": v.Model, "version": v.Version,
				"replaced": replacedAny})
	})
	if err != nil {
		return nil, grpcError(err)
	}
	return toProto(created), nil
}

// Rollback retires the live consumer deployment and restores the one it replaced.
func (s *Server) Rollback(ctx context.Context, req *pb.RollbackRequest) (*pb.Deployment, error) {
	c, err := caller(ctx, auth.ReleasesManage)
	if err != nil {
		return nil, err
	}
	var restored store.Deployment
	err = s.Store.InTenant(ctx, c.Tenant.ID, func(tx pgx.Tx) error {
		all, err := store.Deployments(ctx, tx, c.Tenant.ID, req.GetModelId())
		if err != nil {
			return err
		}
		var current *store.Deployment
		for i := range all {
			d := &all[i]
			if d.Environment != "consumer" {
				continue
			}
			if current == nil && d.Status == "active" {
				current = d
			} else if current != nil && d.Status == "superseded" && d.VersionID != current.VersionID {
				restored = *d
				break
			}
		}
		if current == nil {
			return status.Error(codes.FailedPrecondition, "nothing is live for consumers to roll back")
		}
		if restored.ID == "" {
			return status.Error(codes.FailedPrecondition, "no earlier release to roll back to")
		}
		if _, err := tx.Exec(ctx, `UPDATE deployments SET status = 'rolled_back' WHERE id = $1`,
			current.ID); err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `UPDATE deployments SET status = 'active' WHERE id = $1`,
			restored.ID); err != nil {
			return err
		}
		restored.Status = "active"
		v, err := store.GetVersion(ctx, tx, c.Tenant.ID, current.VersionID)
		if err != nil {
			return err
		}
		if v.State == "released" {
			if err := store.SetVersionState(ctx, tx, v.ID, "rolled_back"); err != nil {
				return err
			}
			if err := audit.Record(ctx, tx, c.Tenant.ID, c.actor(), "version.rolled_back",
				"model_version", v.ID, map[string]any{"from": "released", "to": "rolled_back",
					"reason": "rolled back to v" + restored.Version}); err != nil {
				return err
			}
		}
		return audit.Record(ctx, tx, c.Tenant.ID, c.actor(), "deployment.rolled_back",
			"deployment", current.ID, map[string]any{"model": current.Model,
				"from_version": current.Version, "to_version": restored.Version})
	})
	if err != nil {
		return nil, grpcError(err)
	}
	return toProto(restored), nil
}

// Kill stops every deployment of a version and blocks it (idempotent).
func (s *Server) Kill(ctx context.Context, req *pb.KillRequest) (*pb.KillResponse, error) {
	c, err := caller(ctx, auth.KillSwitch)
	if err != nil {
		return nil, err
	}
	if len(req.GetReason()) < 3 {
		return nil, status.Error(codes.InvalidArgument, "give a reason for using the kill switch")
	}
	response := &pb.KillResponse{}
	err = s.Store.InTenant(ctx, c.Tenant.ID, func(tx pgx.Tx) error {
		v, err := store.GetVersion(ctx, tx, c.Tenant.ID, req.GetVersionId())
		if err != nil {
			return err
		}
		if v.State != "killed" && !slices.Contains(killableStates, v.State) {
			return status.Errorf(codes.FailedPrecondition, "a %s version cannot be killed", v.State)
		}
		tag, err := tx.Exec(ctx, `UPDATE deployments SET status = 'killed'
			WHERE version_id = $1 AND status = 'active'`, v.ID)
		if err != nil {
			return err
		}
		response.DeploymentsStopped = int32(tag.RowsAffected())
		response.VersionState = "killed"
		if v.State == "killed" {
			return nil
		}
		if err := store.SetVersionState(ctx, tx, v.ID, "killed"); err != nil {
			return err
		}
		return audit.Record(ctx, tx, c.Tenant.ID, c.actor(), "version.killed", "model_version",
			v.ID, map[string]any{"from": v.State, "to": "killed", "reason": req.GetReason(),
				"deployments_stopped": int64(tag.RowsAffected())})
	})
	if err != nil {
		return nil, grpcError(err)
	}
	return response, nil
}

// ListDeployments returns the organization's deployments, newest first.
func (s *Server) ListDeployments(ctx context.Context, req *pb.ListDeploymentsRequest) (
	*pb.ListDeploymentsResponse, error) {
	c, err := caller(ctx, auth.ModelsRead)
	if err != nil {
		return nil, err
	}
	out := &pb.ListDeploymentsResponse{}
	err = s.Store.InTenant(ctx, c.Tenant.ID, func(tx pgx.Tx) error {
		all, err := store.Deployments(ctx, tx, c.Tenant.ID, req.GetModelId())
		for _, d := range all {
			out.Deployments = append(out.Deployments, toProto(d))
		}
		return err
	})
	if err != nil {
		return nil, grpcError(err)
	}
	return out, nil
}

// Resolve tells the gateway which version serves a model, if any.
func (s *Server) Resolve(ctx context.Context, req *pb.ResolveRequest) (*pb.ResolveResponse, error) {
	env, permission := "dev", auth.DevEndpoints
	if req.GetEnvironment() == pb.Environment_CONSUMER {
		env, permission = "consumer", auth.ConsumerUse
	}
	c, err := caller(ctx, permission)
	if err != nil {
		return nil, err
	}
	out := &pb.ResolveResponse{}
	err = s.Store.InTenant(ctx, c.Tenant.ID, func(tx pgx.Tx) error {
		d, err := store.ActiveFor(ctx, tx, c.Tenant.ID, req.GetModel(), env)
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		if err != nil {
			return err
		}
		v, err := store.GetVersion(ctx, tx, c.Tenant.ID, d.VersionID)
		if err != nil {
			return err
		}
		out.Found, out.Killed = true, v.State == "killed"
		out.VersionId, out.Version, out.Endpoint = d.VersionID, d.Version, d.Endpoint
		return nil
	})
	if err != nil {
		return nil, grpcError(err)
	}
	return out, nil
}
