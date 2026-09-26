package audit

import "testing"

// Expected values were produced by Python:
//
//	json.dumps(v, sort_keys=True, separators=(",", ":"))
func TestCanonicalMatchesPython(t *testing.T) {
	cases := []struct {
		in   any
		want string
	}{
		{map[string]any{"b": 1, "a": "x"}, `{"a":"x","b":1}`},
		{map[string]any{"s": "Café <&> \"q\" \\ /"}, `{"s":"Caf\u00e9 <&> \"q\" \\ /"}`},
		{map[string]any{"e": "😀", "n": nil, "t": true, "l": []any{"a", int64(2)}},
			`{"e":"\ud83d\ude00","l":["a",2],"n":null,"t":true}`},
		{map[string]any{"c": "line\nnext\ttab\u0001"}, `{"c":"line\nnext\ttab\u0001"}`},
	}
	for _, c := range cases {
		got, err := Canonical(c.in)
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Errorf("Canonical(%v) = %s, want %s", c.in, got, c.want)
		}
	}
	if _, err := Canonical(map[string]any{"f": 1.5}); err == nil {
		t.Error("floats are not canonical here and must be rejected")
	}
}

func TestDigestMatchesPython(t *testing.T) {
	id := "user-1"
	// hashlib.sha256(json.dumps({...}, sort_keys=True, separators=(",", ":")).encode())
	event := Event{Tenant: "t1", Seq: 1, ActorID: &id, Actor: "ann@example.com",
		Action: "deployment.created", TargetType: "deployment", TargetID: "d1",
		Details: map[string]any{"environment": "dev"}, Prev: Genesis}
	got, err := Digest(event)
	if err != nil {
		t.Fatal(err)
	}
	const want = "7162bc181d65d762992898a13feb90de3852dca9b45c3c85d1b0d19afcad6655"
	if got != want {
		t.Errorf("Digest = %s, want %s", got, want)
	}
}
