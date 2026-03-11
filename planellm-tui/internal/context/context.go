// internal/context/context.go
package context

import "github.com/jasperan/planellm-tui/internal/theme"

// Context holds shared services and state injected into all pages.
type Context struct {
	Theme  *theme.Theme
	API    APIClient
	Width  int
	Height int
}

// APIClient defines the planeLLM FastAPI service interface.
type APIClient interface {
	GetStatus() (*SystemStatus, error)
	ListFiles() (*FileList, error)
	GenerateTopic(topic string) (*TopicResult, error)
	CreateTranscript(contentFile string, detailed bool) (*TranscriptResult, error)
	GenerateAudio(transcriptFile, ttsModel, fishRef, fishEmotion string) (*AudioResult, error)
}

// Data types — flat and simple.

// SystemStatus reflects health of backend services.
type SystemStatus struct {
	OCIConfig      bool `json:"oci_config"`
	FFmpeg         bool `json:"ffmpeg"`
	FishSDK        bool `json:"fish_sdk"`
	ResourcesCount int  `json:"resources_count"`
}

// FileList categorizes available resource files.
type FileList struct {
	Content     []string `json:"content"`
	Transcripts []string `json:"transcripts"`
	Audio       []string `json:"audio"`
}

// TopicResult is the output of topic exploration.
type TopicResult struct {
	Success        bool     `json:"success"`
	Message        string   `json:"message"`
	QuestionsFile  string   `json:"questions_file"`
	ContentFile    string   `json:"content_file"`
	Questions      []string `json:"questions"`
	ContentPreview string   `json:"content_preview"`
}

// TranscriptResult is the output of transcript creation.
type TranscriptResult struct {
	Success           bool   `json:"success"`
	Message           string `json:"message"`
	TranscriptFile    string `json:"transcript_file"`
	TranscriptPreview string `json:"transcript_preview"`
	Length            int    `json:"length"`
}

// AudioResult is the output of audio generation.
type AudioResult struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	AudioFile string `json:"audio_file"`
	AudioPath string `json:"audio_path"`
}
