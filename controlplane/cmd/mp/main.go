// Command mp is the MP command-line tool for deployments and the kill switch.
//
//	mp login   --api https://usps.mp.com --email you@usps.com
//	mp status  [--model <id>]
//	mp deploy  --version <id> --env dev|consumer
//	mp rollback --model <id>
//	mp kill    --version <id> --reason "leak reported"
//
// The session token is saved in ~/.config/mp/token (readable only by you). Every command
// needs --org (or MP_ORG) and talks to the control plane at --controlplane (or MP_CONTROLPLANE).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"golang.org/x/term"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	pb "github.com/r7bb/Model-Passport/controlplane/gen/mp/controlplane/v1"
)

const usage = `mp: deployments, rollback, and the kill switch for MP

  mp login    --api <url> --email <email>
  mp status   [--model <id>]
  mp deploy   --version <id> --env dev|consumer
  mp rollback --model <id>
  mp kill     --version <id> --reason <text>

Common flags: --org <slug> (MP_ORG), --controlplane <host:port> (MP_CONTROLPLANE)`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, usage)
		os.Exit(2)
	}
	if err := run(os.Args[1], os.Args[2:]); err != nil {
		if s, ok := status.FromError(err); ok {
			err = errors.New(s.Message())
		}
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func tokenPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		dir = os.TempDir()
	}
	return filepath.Join(dir, "mp", "token")
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func run(command string, args []string) error {
	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	org := flags.String("org", env("MP_ORG", ""), "organization slug")
	addr := flags.String("controlplane", env("MP_CONTROLPLANE", "localhost:9090"), "control plane address")
	api := flags.String("api", env("MP_API", ""), "platform API URL (login)")
	email := flags.String("email", "", "email address (login)")
	model := flags.String("model", "", "model id")
	version := flags.String("version", "", "model version id")
	environment := flags.String("env", "", "dev or consumer")
	reason := flags.String("reason", "", "why the kill switch is used")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if command == "login" {
		return login(*api, *email)
	}
	if *org == "" {
		return errors.New("set --org or MP_ORG to the organization's slug")
	}
	token, err := os.ReadFile(tokenPath())
	if err != nil {
		return errors.New("not signed in: run `mp login` first")
	}
	conn, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pb.NewControlPlaneClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ctx = metadata.AppendToOutgoingContext(ctx, "authorization",
		"Bearer "+strings.TrimSpace(string(token)), "x-mp-tenant", *org)
	switch command {
	case "status":
		out, err := client.ListDeployments(ctx, &pb.ListDeploymentsRequest{ModelId: *model})
		if err != nil {
			return err
		}
		table := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(table, "MODEL\tVERSION\tENVIRONMENT\tSTATUS\tENDPOINT\tSINCE")
		for _, d := range out.GetDeployments() {
			fmt.Fprintf(table, "%s\t%s\t%s\t%s\t%s\t%s\n", d.GetModel(), d.GetVersion(),
				strings.ToLower(d.GetEnvironment().String()), d.GetStatus(), d.GetEndpoint(), d.GetCreatedAt())
		}
		return table.Flush()
	case "deploy":
		target, ok := map[string]pb.Environment{"dev": pb.Environment_DEV, "consumer": pb.Environment_CONSUMER}[*environment]
		if !ok || *version == "" {
			return errors.New("usage: mp deploy --version <id> --env dev|consumer")
		}
		d, err := client.Deploy(ctx, &pb.DeployRequest{VersionId: *version, Environment: target})
		if err != nil {
			return err
		}
		fmt.Printf("deployed %s v%s to %s at %s\n", d.GetModel(), d.GetVersion(), *environment, d.GetEndpoint())
	case "rollback":
		d, err := client.Rollback(ctx, &pb.RollbackRequest{ModelId: *model})
		if err != nil {
			return err
		}
		fmt.Printf("rolled back %s to v%s\n", d.GetModel(), d.GetVersion())
	case "kill":
		out, err := client.Kill(ctx, &pb.KillRequest{VersionId: *version, Reason: *reason})
		if err != nil {
			return err
		}
		fmt.Printf("kill switch on: version %s, %d deployment(s) stopped\n", out.GetVersionState(), out.GetDeploymentsStopped())
	default:
		return fmt.Errorf("unknown command %q\n\n%s", command, usage)
	}
	return nil
}

func login(api, email string) error {
	if api == "" || email == "" {
		return errors.New("usage: mp login --api <url> --email <email>")
	}
	fmt.Fprint(os.Stderr, "password: ")
	password, err := term.ReadPassword(int(os.Stdin.Fd()))
	fmt.Fprintln(os.Stderr)
	if err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]string{"email": email, "password": string(password)})
	response, err := http.Post(strings.TrimRight(api, "/")+"/api/v1/auth/login", "application/json",
		bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var out struct {
		AccessToken string `json:"access_token"`
		Detail      string `json:"detail"`
	}
	if err := json.NewDecoder(response.Body).Decode(&out); err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("sign-in failed: %s", out.Detail)
	}
	path := tokenPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	if err := os.WriteFile(path, []byte(out.AccessToken), 0o600); err != nil {
		return err
	}
	fmt.Println("signed in")
	return nil
}
