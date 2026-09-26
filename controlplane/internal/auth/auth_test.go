package auth

import (
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func token(t *testing.T, secret string, exp time.Time, method jwt.SigningMethod) string {
	t.Helper()
	claims := Claims{SuperAdmin: false, RegisteredClaims: jwt.RegisteredClaims{
		Subject: "user-1", ExpiresAt: jwt.NewNumericDate(exp)}}
	signed, err := jwt.NewWithClaims(method, claims).SignedString([]byte(secret))
	if err != nil {
		t.Fatal(err)
	}
	return signed
}

func TestVerify(t *testing.T) {
	secret := "s-3cret-that-is-long-enough-for-hs256-use"
	good := token(t, secret, time.Now().Add(time.Hour), jwt.SigningMethodHS256)
	claims, err := Verify("Bearer "+good, secret)
	if err != nil || claims.Subject != "user-1" {
		t.Fatalf("valid token rejected: %v", err)
	}
	for name, bad := range map[string]string{
		"wrong secret": token(t, "other-secret-other-secret-other-secret", time.Now().Add(time.Hour), jwt.SigningMethodHS256),
		"expired":      token(t, secret, time.Now().Add(-time.Minute), jwt.SigningMethodHS256),
		"empty":        "",
	} {
		if _, err := Verify(bad, secret); err == nil {
			t.Errorf("%s: token accepted", name)
		}
	}
}

func TestRolesMatchThePythonBackend(t *testing.T) {
	cases := []struct {
		role, permission string
		want             bool
	}{
		{"compliance_auditor", KillSwitch, true},
		{"org_admin", ReleasesManage, true},
		{"ml_engineer", ReleasesManage, false},
		{"ml_engineer", DevEndpoints, true},
		{"canary_tester", DevEndpoints, true},
		{"end_consumer", ConsumerUse, true},
		{"end_consumer", ModelsRead, false},
		{"external_reviewer", KillSwitch, false},
		{"", ModelsRead, false},
	}
	for _, c := range cases {
		if got := Allowed(c.role, c.permission, false); got != c.want {
			t.Errorf("Allowed(%q, %q) = %v, want %v", c.role, c.permission, got, c.want)
		}
	}
	if !Allowed("", KillSwitch, true) {
		t.Error("super admins may do anything")
	}
}

func TestTenantFromHost(t *testing.T) {
	for host, want := range map[string]string{
		"usps.mp.com": "usps", "UPS.MP.COM:443": "ups", "mp.com": "", "a.b.mp.com": "",
		"usps.evil.com": "",
	} {
		if got := TenantFromHost(host, "mp.com"); got != want {
			t.Errorf("TenantFromHost(%q) = %q, want %q", host, got, want)
		}
	}
}
