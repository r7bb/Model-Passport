// Command mp-gateway is the API gateway in front of the backend and model servers.
//
//	MP_BACKEND_URL        the FastAPI backend (default http://localhost:8080)
//	MP_INFERENCE_URL      model servers (optional until serving is configured)
//	MP_CONTROLPLANE_ADDR  the control plane (default localhost:9090)
//	MP_BASE_DOMAIN        organizations are <slug>.<base domain>
//	MP_JWT_SECRET         the backend's session-signing secret
//	MP_GATEWAY_ADDR       listen address (default :8000)
package main

import (
	"log"
	"net/http"
	"net/url"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
	"github.com/r7bb/Model-Passport/controlplane/internal/config"
	"github.com/r7bb/Model-Passport/controlplane/internal/gateway"
)

func main() {
	secret, err := config.Secret()
	if err != nil {
		log.Fatal(err)
	}
	backend, err := url.Parse(config.Env("MP_BACKEND_URL", "http://localhost:8080"))
	if err != nil {
		log.Fatal(err)
	}
	var inference *url.URL
	if raw := config.Env("MP_INFERENCE_URL", ""); raw != "" {
		if inference, err = url.Parse(raw); err != nil {
			log.Fatal(err)
		}
	}
	// Inside the cluster the control plane is reached over the private network; put mTLS
	// (a service mesh) in front of it in production.
	conn, err := grpc.NewClient(config.Env("MP_CONTROLPLANE_ADDR", "localhost:9090"),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	handler := gateway.New(gateway.Config{Backend: backend, Inference: inference,
		BaseDomain: config.Env("MP_BASE_DOMAIN", "mp.localhost"), Secret: secret,
		Resolver: pb.NewControlPlaneClient(conn)})
	server := &http.Server{Addr: config.Env("MP_GATEWAY_ADDR", ":8000"), Handler: handler,
		ReadHeaderTimeout: 10 * time.Second}
	log.Printf("gateway listening on %s", server.Addr)
	log.Fatal(server.ListenAndServe())
}
