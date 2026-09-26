package audit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"

	"github.com/jackc/pgx/v5"
)

// Genesis is the previous-hash of an organization's first event.
const Genesis = "0000000000000000000000000000000000000000000000000000000000000000"

// Actor is who made a change: a user (ID and email) or a service ("controlplane").
type Actor struct {
	ID    *string
	Label string
}

// Event is one audit log entry, as hashed by Digest.
type Event struct {
	Tenant     string
	Seq        int64
	ActorID    *string
	Actor      string
	Action     string
	TargetType string
	TargetID   string
	Details    map[string]any
	Prev       string
}

// Digest is the SHA256 of the event's canonical JSON, matching the Python backend.
func Digest(e Event) (string, error) {
	var actorID any
	if e.ActorID != nil {
		actorID = *e.ActorID
	}
	details := e.Details
	if details == nil {
		details = map[string]any{}
	}
	body, err := Canonical(map[string]any{
		"tenant": e.Tenant, "seq": e.Seq, "actor_id": actorID, "actor": e.Actor,
		"action": e.Action, "target_type": e.TargetType, "target_id": e.TargetID,
		"details": details, "prev": e.Prev,
	})
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256([]byte(body))
	return hex.EncodeToString(sum[:]), nil
}

// Record appends an event in tx. The organization's chain is locked for the transaction
// (pg_advisory_xact_lock on hashtext(tenant), as in Python), so writers never fork it.
func Record(ctx context.Context, tx pgx.Tx, tenant string, actor Actor, action, targetType,
	targetID string, details map[string]any) error {
	if _, err := tx.Exec(ctx, "SELECT pg_advisory_xact_lock(hashtext($1))", tenant); err != nil {
		return err
	}
	event := Event{Tenant: tenant, ActorID: actor.ID, Actor: actor.Label, Action: action,
		TargetType: targetType, TargetID: targetID, Details: details, Prev: Genesis, Seq: 1}
	var lastSeq int64
	var lastHash string
	err := tx.QueryRow(ctx, `SELECT seq, hash FROM audit_events WHERE tenant_id = $1
		ORDER BY seq DESC LIMIT 1`, tenant).Scan(&lastSeq, &lastHash)
	switch {
	case err == nil:
		event.Seq, event.Prev = lastSeq+1, lastHash
	case !errors.Is(err, pgx.ErrNoRows):
		return err
	}
	hash, err := Digest(event)
	if err != nil {
		return err
	}
	encoded, err := json.Marshal(event.Details)
	if err != nil {
		return err
	}
	_, err = tx.Exec(ctx, `INSERT INTO audit_events (tenant_id, seq, actor_id, actor, action,
		target_type, target_id, details, prev_hash, hash, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8::json, $9, $10, now())`,
		tenant, event.Seq, event.ActorID, event.Actor, event.Action, event.TargetType,
		event.TargetID, string(encoded), event.Prev, hash)
	return err
}
