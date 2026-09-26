// Package config reads the MP_* environment variables shared with the Python backend.
package config

import (
	"fmt"
	"os"
)

// Env returns the variable or a default.
func Env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

// Secret returns MP_JWT_SECRET, which must be at least 32 characters.
func Secret() (string, error) {
	secret := os.Getenv("MP_JWT_SECRET")
	if len(secret) < 32 {
		return "", fmt.Errorf("set MP_JWT_SECRET to at least 32 characters (the backend's value)")
	}
	return secret, nil
}
