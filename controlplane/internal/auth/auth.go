// Package auth checks session tokens and roles, mirroring the Python backend's rules
// (model_passport.platform.rbac) so a caller has the same rights through any door.
package auth

import (
	"errors"
	"fmt"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

// Permission names match the Python backend.
const (
	ModelsRead      = "models.read"
	DevEndpoints    = "endpoints.dev"
	ConsumerUse     = "endpoints.consumer"
	ReleasesManage  = "releases.manage"
	KillSwitch      = "killswitch.activate"
	ApprovalsDecide = "approvals.decide"
)

var rolePermissions = map[string]map[string]bool{
	"org_admin":          set(ModelsRead, ReleasesManage, KillSwitch, ApprovalsDecide),
	"ml_engineer":        set(ModelsRead, DevEndpoints),
	"compliance_auditor": set(ModelsRead, KillSwitch, ApprovalsDecide),
	"canary_tester":      set(ModelsRead, DevEndpoints),
	"end_consumer":       set(ConsumerUse),
	"external_reviewer":  set(),
}

func set(items ...string) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, item := range items {
		out[item] = true
	}
	return out
}

// Allowed reports whether role (empty for none) may use permission; super admins may do all.
func Allowed(role, permission string, superAdmin bool) bool {
	if superAdmin {
		return true
	}
	return rolePermissions[role][permission]
}

// Claims are the session token's contents, as issued by the Python backend.
type Claims struct {
	SuperAdmin bool `json:"sa"`
	jwt.RegisteredClaims
}

// ErrUnauthenticated means the token is missing, malformed, expired, or forged.
var ErrUnauthenticated = errors.New("sign in first: missing, expired, or invalid session")

// Verify checks an HS256 token (with or without a "Bearer " prefix) and returns its claims.
func Verify(token, secret string) (*Claims, error) {
	token = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(token), "Bearer "))
	if token == "" {
		return nil, ErrUnauthenticated
	}
	claims := &Claims{}
	parsed, err := jwt.ParseWithClaims(token, claims, func(t *jwt.Token) (any, error) {
		if t.Method != jwt.SigningMethodHS256 {
			return nil, fmt.Errorf("unexpected signing method %v", t.Header["alg"])
		}
		return []byte(secret), nil
	}, jwt.WithExpirationRequired())
	if err != nil || !parsed.Valid || claims.Subject == "" {
		return nil, ErrUnauthenticated
	}
	return claims, nil
}

// TenantFromHost returns "usps" for host "usps.mp.com" with base domain "mp.com".
func TenantFromHost(host, baseDomain string) string {
	host = strings.ToLower(strings.Split(host, ":")[0])
	suffix := "." + strings.ToLower(baseDomain)
	if baseDomain == "" || !strings.HasSuffix(host, suffix) {
		return ""
	}
	slug := strings.TrimSuffix(host, suffix)
	if strings.Contains(slug, ".") {
		return "" // only one level: usps.mp.com, not a.b.mp.com
	}
	return slug
}
