// Command mp-controlplane serves the ControlPlane gRPC API.
//
//	MP_DATABASE_URL   the platform's PostgreSQL database (a non-superuser role)
//	MP_JWT_SECRET     the backend's session-signing secret
//	MP_CONTROLPLANE_ADDR  listen address (default :9090)
package main

import (
	"context"
	"log"
	"net"
	"os/signal"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
	"github.com/r7bb/Model-Passport/controlplane/internal/config"
	"github.com/r7bb/Model-Passport/controlplane/internal/server"
	"github.com/r7bb/Model-Passport/controlplane/internal/store"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	secret, err := config.Secret()
	if err != nil {
		log.Fatal(err)
	}
	db, err := store.Open(ctx, config.Env("MP_DATABASE_URL", ""))
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()
	srv := &server.Server{Store: db, Secret: secret}
	grpcServer := grpc.NewServer(grpc.UnaryInterceptor(srv.Authenticate))
	pb.RegisterControlPlaneServer(grpcServer, srv)
	healthpb.RegisterHealthServer(grpcServer, health.NewServer())
	addr := config.Env("MP_CONTROLPLANE_ADDR", ":9090")
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatal(err)
	}
	go func() {
		<-ctx.Done()
		grpcServer.GracefulStop()
	}()
	log.Printf("control plane listening on %s", addr)
	if err := grpcServer.Serve(listener); err != nil {
		log.Fatal(err)
	}
}
