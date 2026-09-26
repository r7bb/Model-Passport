// Package store is the control plane's access to the platform database.
//
// Every transaction sets app.tenant_id, exactly as the Python backend does, so PostgreSQL
// row-level security confines the control plane to one organization at a time.
package store

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// AllTenants lets platform-wide lookups (finding an organization by slug) see every row.
const AllTenants = "*"

// ErrNotFound means the row does not exist in this organization.
var ErrNotFound = errors.New("not found")

// Store wraps a connection pool.
type Store struct {
	pool *pgxpool.Pool
}

// Open connects using a PostgreSQL URL; SQLAlchemy-style "postgresql+psycopg://" is accepted.
func Open(ctx context.Context, url string) (*Store, error) {
	for _, prefix := range []string{"postgresql+psycopg://", "postgresql+psycopg2://"} {
		url = strings.Replace(url, prefix, "postgresql://", 1)
	}
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		return nil, err
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("database unreachable: %w", err)
	}
	return &Store{pool: pool}, nil
}

// Close releases the pool.
func (s *Store) Close() { s.pool.Close() }

// InTenant runs fn in a transaction limited to tenant (or AllTenants), committing on success.
func (s *Store) InTenant(ctx context.Context, tenant string, fn func(pgx.Tx) error) error {
	return pgx.BeginFunc(ctx, s.pool, func(tx pgx.Tx) error {
		if _, err := tx.Exec(ctx, "SELECT set_config('app.tenant_id', $1, true)", tenant); err != nil {
			return err
		}
		return fn(tx)
	})
}

// Tenant is an organization.
type Tenant struct {
	ID     string
	Slug   string
	Status string
}

// Member returns the organization for slug and the user's role in it ("" if not a member).
func (s *Store) Member(ctx context.Context, slug, userID string) (Tenant, string, error) {
	var t Tenant
	err := s.InTenant(ctx, AllTenants, func(tx pgx.Tx) error {
		err := tx.QueryRow(ctx, "SELECT id, slug, status FROM tenants WHERE slug = $1", slug).
			Scan(&t.ID, &t.Slug, &t.Status)
		if errors.Is(err, pgx.ErrNoRows) {
			return ErrNotFound
		}
		return err
	})
	if err != nil {
		return t, "", err
	}
	var role string
	err = s.InTenant(ctx, t.ID, func(tx pgx.Tx) error {
		err := tx.QueryRow(ctx, `SELECT role FROM memberships WHERE tenant_id = $1 AND user_id = $2`,
			t.ID, userID).Scan(&role)
		if errors.Is(err, pgx.ErrNoRows) {
			return nil
		}
		return err
	})
	return t, role, err
}

// UserEmail returns a user's email for audit records.
func (s *Store) UserEmail(ctx context.Context, userID string) (string, error) {
	var email string
	err := s.InTenant(ctx, AllTenants, func(tx pgx.Tx) error {
		return tx.QueryRow(ctx, "SELECT email FROM users WHERE id = $1 AND NOT disabled", userID).
			Scan(&email)
	})
	if errors.Is(err, pgx.ErrNoRows) {
		return "", ErrNotFound
	}
	return email, err
}

// Version is the part of a model version the control plane needs.
type Version struct {
	ID      string
	ModelID string
	Model   string
	Version string
	State   string
}

// GetVersion loads a version with its model name, locking the row for this transaction.
func GetVersion(ctx context.Context, tx pgx.Tx, tenant, versionID string) (Version, error) {
	var v Version
	err := tx.QueryRow(ctx, `SELECT v.id, v.model_id, m.name, v.version, v.state
		FROM model_versions v JOIN models m ON m.id = v.model_id
		WHERE v.id = $1 AND v.tenant_id = $2 FOR UPDATE OF v`, versionID, tenant).
		Scan(&v.ID, &v.ModelID, &v.Model, &v.Version, &v.State)
	if errors.Is(err, pgx.ErrNoRows) {
		return v, ErrNotFound
	}
	return v, err
}

// SetVersionState changes a version's lifecycle state.
func SetVersionState(ctx context.Context, tx pgx.Tx, versionID, state string) error {
	_, err := tx.Exec(ctx, `UPDATE model_versions SET state = $1, updated_at = now() WHERE id = $2`,
		state, versionID)
	return err
}

// Deployment is one version serving in one environment.
type Deployment struct {
	ID          string
	VersionID   string
	ModelID     string
	Model       string
	Version     string
	Environment string
	Status      string
	Endpoint    string
	CreatedAt   time.Time
}

const deploymentColumns = `d.id, d.version_id, v.model_id, m.name, v.version, d.environment,
	d.status, d.endpoint, d.created_at`

func scanDeployments(rows pgx.Rows) ([]Deployment, error) {
	return pgx.CollectRows(rows, func(row pgx.CollectableRow) (Deployment, error) {
		var d Deployment
		err := row.Scan(&d.ID, &d.VersionID, &d.ModelID, &d.Model, &d.Version, &d.Environment,
			&d.Status, &d.Endpoint, &d.CreatedAt)
		return d, err
	})
}

// Deployments lists an organization's deployments, newest first (optionally for one model).
func Deployments(ctx context.Context, tx pgx.Tx, tenant, modelID string) ([]Deployment, error) {
	rows, err := tx.Query(ctx, `SELECT `+deploymentColumns+` FROM deployments d
		JOIN model_versions v ON v.id = d.version_id JOIN models m ON m.id = v.model_id
		WHERE d.tenant_id = $1 AND ($2 = '' OR v.model_id = $2)
		ORDER BY d.created_at DESC`, tenant, modelID)
	if err != nil {
		return nil, err
	}
	return scanDeployments(rows)
}

// ActiveFor returns the active deployment of a model (by name) in an environment.
func ActiveFor(ctx context.Context, tx pgx.Tx, tenant, model, env string) (Deployment, error) {
	rows, err := tx.Query(ctx, `SELECT `+deploymentColumns+` FROM deployments d
		JOIN model_versions v ON v.id = d.version_id JOIN models m ON m.id = v.model_id
		WHERE d.tenant_id = $1 AND m.name = $2 AND d.environment = $3 AND d.status = 'active'
		ORDER BY d.created_at DESC LIMIT 1`, tenant, model, env)
	if err != nil {
		return Deployment{}, err
	}
	found, err := scanDeployments(rows)
	if err != nil {
		return Deployment{}, err
	}
	if len(found) == 0 {
		return Deployment{}, ErrNotFound
	}
	return found[0], nil
}
