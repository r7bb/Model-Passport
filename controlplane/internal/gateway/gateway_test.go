package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"google.golang.org/grpc"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
	"github.com/r7bb/Model-Passport/controlplane/internal/auth"
)

const secret = "gateway-test-secret-gateway-test-secret"

type fakeResolver struct{ response *pb.ResolveResponse }

func (f fakeResolver) Resolve(context.Context, *pb.ResolveRequest, ...grpc.CallOption) (*pb.ResolveResponse, error) {
	return f.response, nil
}

func signed(t *testing.T) string {
	t.Helper()
	claims := auth.Claims{RegisteredClaims: jwt.RegisteredClaims{Subject: "u1",
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour))}}
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(secret))
	if err != nil {
		t.Fatal(err)
	}
	return "Bearer " + token
}

func setup(t *testing.T, resolved *pb.ResolveResponse) (http.Handler, *httptest.Server) {
	t.Helper()
	echo := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"tenant": r.Header.Get("X-MP-Tenant"), "path": r.URL.Path,
			"version": r.Header.Get("X-MP-Version")})
	}))
	t.Cleanup(echo.Close)
	target, _ := url.Parse(echo.URL)
	return New(Config{Backend: target, Inference: target, BaseDomain: "mp.test", Secret: secret,
		Resolver: fakeResolver{resolved}}), echo
}

func request(h http.Handler, host, path, token, spoof string) (*httptest.ResponseRecorder, map[string]string) {
	r := httptest.NewRequest(http.MethodGet, path, nil)
	r.Host = host
	if token != "" {
		r.Header.Set("Authorization", token)
	}
	if spoof != "" {
		r.Header.Set("X-MP-Tenant", spoof)
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	body := map[string]string{}
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	return w, body
}

func TestTenantComesFromTheSubdomainNotTheClient(t *testing.T) {
	h, _ := setup(t, nil)
	w, body := request(h, "usps.mp.test", "/api/v1/models", signed(t), "ups")
	if w.Code != http.StatusOK || body["tenant"] != "usps" {
		t.Fatalf("got %d %v; the spoofed X-MP-Tenant must be replaced by the subdomain", w.Code, body)
	}
	_, body = request(h, "mp.test", "/api/v1/platform/tenants", signed(t), "ups")
	if body["tenant"] != "" {
		t.Fatalf("no subdomain means no tenant, even if the client sends one: %v", body)
	}
}

func TestLoginIsCheckedAtTheEdge(t *testing.T) {
	h, _ := setup(t, nil)
	if w, _ := request(h, "usps.mp.test", "/api/v1/models", "", ""); w.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated request forwarded: %d", w.Code)
	}
	if w, _ := request(h, "usps.mp.test", "/api/v1/models", "Bearer forged", ""); w.Code != http.StatusUnauthorized {
		t.Fatalf("forged token forwarded: %d", w.Code)
	}
	if w, _ := request(h, "usps.mp.test", "/api/v1/auth/login", "", ""); w.Code != http.StatusOK {
		t.Fatalf("sign-in must stay public: %d", w.Code)
	}
}

func TestModelEndpointsHonorTheKillSwitch(t *testing.T) {
	live, _ := setup(t, &pb.ResolveResponse{Found: true, VersionId: "v2"})
	w, body := request(live, "usps.mp.test", "/v1/consumer/assistant/generate", signed(t), "")
	if w.Code != http.StatusOK || body["version"] != "v2" {
		t.Fatalf("live model not served: %d %v", w.Code, body)
	}
	killed, _ := setup(t, &pb.ResolveResponse{Found: true, Killed: true, VersionId: "v2"})
	if w, _ := request(killed, "usps.mp.test", "/v1/consumer/assistant/generate", signed(t), ""); w.Code != http.StatusLocked {
		t.Fatalf("killed model served: %d", w.Code)
	}
	missing, _ := setup(t, &pb.ResolveResponse{Found: false})
	if w, _ := request(missing, "usps.mp.test", "/v1/dev/assistant", signed(t), ""); w.Code != http.StatusNotFound {
		t.Fatalf("undeployed model served: %d", w.Code)
	}
}
