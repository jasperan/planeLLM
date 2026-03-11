package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	ctx "github.com/jasperan/planellm-tui/internal/context"
)

// Compile-time interface check.
var _ ctx.APIClient = (*APIClient)(nil)

// APIClient implements ctx.APIClient over HTTP.
type APIClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewAPIClient creates an HTTP client for the planeLLM API.
func NewAPIClient(baseURL string) *APIClient {
	return &APIClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: &http.Client{Timeout: 300 * time.Second},
	}
}

func (c *APIClient) get(path string, result interface{}) error {
	resp, err := c.httpClient.Get(c.baseURL + path)
	if err != nil {
		return fmt.Errorf("GET %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: status %d", path, resp.StatusCode)
	}
	return json.NewDecoder(resp.Body).Decode(result)
}

func (c *APIClient) post(path string, body interface{}, result interface{}) error {
	var reqBody *bytes.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(b)
	} else {
		reqBody = bytes.NewReader([]byte("{}"))
	}
	resp, err := c.httpClient.Post(c.baseURL+path, "application/json", reqBody)
	if err != nil {
		return fmt.Errorf("POST %s: %w", path, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("POST %s: status %d", path, resp.StatusCode)
	}
	if result != nil {
		return json.NewDecoder(resp.Body).Decode(result)
	}
	return nil
}

// GetStatus fetches the system health status.
func (c *APIClient) GetStatus() (*ctx.SystemStatus, error) {
	var s ctx.SystemStatus
	return &s, c.get("/api/status", &s)
}

// ListFiles lists available resource files by category.
func (c *APIClient) ListFiles() (*ctx.FileList, error) {
	var f ctx.FileList
	return &f, c.get("/api/files", &f)
}

// GenerateTopic generates educational content from a topic.
func (c *APIClient) GenerateTopic(topic string) (*ctx.TopicResult, error) {
	var r ctx.TopicResult
	return &r, c.post("/api/topic/generate", map[string]string{"topic": topic}, &r)
}

// CreateTranscript creates a podcast transcript from content.
func (c *APIClient) CreateTranscript(contentFile string, detailed bool) (*ctx.TranscriptResult, error) {
	var r ctx.TranscriptResult
	body := map[string]interface{}{
		"content_file": contentFile,
		"detailed":     detailed,
	}
	return &r, c.post("/api/transcript/create", body, &r)
}

// GenerateAudio generates podcast audio from a transcript.
func (c *APIClient) GenerateAudio(transcriptFile, ttsModel, fishRef, fishEmotion string) (*ctx.AudioResult, error) {
	var r ctx.AudioResult
	body := map[string]string{
		"transcript_file":   transcriptFile,
		"tts_model":         ttsModel,
		"fish_reference": fishRef,
		"fish_emotion":      fishEmotion,
	}
	return &r, c.post("/api/audio/generate", body, &r)
}
