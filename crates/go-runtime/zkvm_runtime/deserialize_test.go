package zkvm_runtime

import (
	"reflect"
	"testing"
)

type nestedPayload struct {
	Value *uint32
}

func TestDeserializeDataRejectsTruncatedInput(t *testing.T) {
	var out uint16
	_, err := deserializeData([]byte{0x01}, reflect.ValueOf(&out).Elem(), 0)
	if err == nil {
		t.Fatal("expected an error for truncated input")
	}
}

func TestDeserializeDataRejectsOversizedSliceLength(t *testing.T) {
	var out []byte
	data := []byte{
		0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff,
	}
	_, err := deserializeData(data, reflect.ValueOf(&out).Elem(), 0)
	if err == nil {
		t.Fatal("expected an error for oversized slice length")
	}
}

func TestDeserializeDataAllocatesNilPointerField(t *testing.T) {
	var out nestedPayload
	data := []byte{
		0x01,
		0x78, 0x56, 0x34, 0x12,
	}
	index, err := deserializeData(data, reflect.ValueOf(&out).Elem(), 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if index != len(data) {
		t.Fatalf("unexpected index: got %d want %d", index, len(data))
	}
	if out.Value == nil {
		t.Fatal("expected nil pointer field to be allocated")
	}
	if *out.Value != 0x12345678 {
		t.Fatalf("unexpected pointer value: got %x want %x", *out.Value, 0x12345678)
	}
}
