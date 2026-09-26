// Package gateway is the API gateway: subdomain -> organization, and a login check.
//
// Requests to "<slug>.<base domain>" reach the backend with X-MP-Tenant set from the host.
// Any X-MP-Tenant sent by the client is removed first, so one organization can never claim to
// be another. Every API call except sign-in and the docs needs a valid session before it is
// forwarded. Model endpoints ("/v1/<dev|consumer>/<model>/...") are resolved through the
// control plane, and a version stopped by the kill switch is refused at the edge.
package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
	"github.com/r7bb/Model-Passport/controlplane/internal/auth"
)

// Public paths need no session.
var public = []string{"/health", "/docs", "/openapi.json", "/api/v1/auth/login", "/redoc"}

// Resolver finds what serves a model; the control plane's gRPC client satisfies it.
type Resolver interface {
	Resolve(ctx context.Context, in *pb.ResolveRequest, opts ...grpc.CallOption) (*pb.ResolveResponse, error)
}

// Config wires the gateway.
type Config struct {
	Backend    *url.URL // FastAPI backend
	Inference  *url.URL // model servers (nil until serving is configured)
	BaseDomain string
	Secret     string
	Resolver   Resolver
}

// New returns the gateway handler.
func New(cfg Config) http.Handler {
	backend := httputil.NewSingleHostReverseProxy(cfg.Backend)
	var inference *httputil.ReverseProxy
	if cfg.Inference != nil {
		inference = httputil.NewSingleHostReverseProxy(cfg.Inference)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Header.Del("X-MP-Tenant") // never trust the client's claim
		slug := auth.TenantFromHost(r.Host, cfg.BaseDomain)
		if slug != "" {
			r.Header.Set("X-MP-Tenant", slug)
		}
		if !isPublic(r.URL.Path) {
			if _, err := auth.Verify(r.Header.Get("Authorization"), cfg.Secret); err != nil {
				fail(w, http.StatusUnauthorized, err.Error())
				return
			}
		}
		if strings.HasPrefix(r.URL.Path, "/v1/") {
			serveModel(w, r, cfg, slug, inference)
			return
		}
		backend.ServeHTTP(w, r)
	})
}

func isPublic(path string) bool {
	for _, p := range public {
		if path == p || strings.HasPrefix(path, p+"/") {
			return true
		}
	}
	return false
}

func fail(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"detail": message})
}

// serveModel handles /v1/<env>/<model>/<rest>.
func serveModel(w http.ResponseWriter, r *http.Request, cfg Config, slug string,
	inference *httputil.ReverseProxy) {
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/v1/"), "/", 3)
	if slug == "" || len(parts) < 2 {
		fail(w, http.StatusNotFound, "use https://<organization>.<domain>/v1/<dev|consumer>/<model>")
		return
	}
	env := pb.Environment_DEV
	switch parts[0] {
	case "dev":
	case "consumer":
		env = pb.Environment_CONSUMER
	default:
		fail(w, http.StatusNotFound, "environment must be dev or consumer")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", r.Header.Get("Authorization"),
		"x-mp-tenant", slug)
	resolved, err := cfg.Resolver.Resolve(ctx, &pb.ResolveRequest{Model: parts[1], Environment: env})
	switch {
	case err != nil:
		fail(w, http.StatusForbidden, "not allowed to use this model: "+err.Error())
	case !resolved.GetFound():
		fail(w, http.StatusNotFound, "no version of this model is deployed there")
	case resolved.GetKilled():
		fail(w, http.StatusLocked, "this model version is blocked by the kill switch")
	case inference == nil:
		fail(w, http.StatusNotImplemented, "model serving is not configured")
	default:
		r.Header.Set("X-MP-Version", resolved.GetVersionId())
		inference.ServeHTTP(w, r)
	}
}
