package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const baseURL = "http://localhost:8080/api/v1"

var (
	authToken     string
	lastSessionID string
	reader        = bufio.NewReader(os.Stdin)
	client        = &http.Client{Timeout: 30 * time.Second}
)

func main() {
	fmt.Println("Welcome to FreeChat CLI")
	for {
		if authToken == "" {
			printAuthMenu()
		} else {
			printMainMenu()
		}
	}
}

func printAuthMenu() {
	fmt.Println("\n=== Auth Menu ===")
	fmt.Println("1. Login")
	fmt.Println("2. Register")
	fmt.Println("3. Exit")
	fmt.Print("> ")

	choice, err := reader.ReadString('\n')
	if err != nil {
		os.Exit(0)
	}
	choice = strings.TrimSpace(choice)

	switch choice {
	case "1":
		handleLogin()
	case "2":
		handleRegister()
	case "3":
		fmt.Println("Goodbye!")
		os.Exit(0)
	default:
		fmt.Println("Invalid choice")
	}
}

func printMainMenu() {
	fmt.Println("\n=== Main Menu ===")
	fmt.Println("1. Start New Chat")
	if lastSessionID != "" {
		fmt.Printf("2. Resume Chat (%s)\n", lastSessionID)
	} else {
		fmt.Println("2. Resume Chat (No recent session)")
	}
	fmt.Println("3. View History")
	fmt.Println("4. Logout")
	fmt.Println("5. Exit")
	fmt.Print("> ")

	choice, err := reader.ReadString('\n')
	if err != nil {
		os.Exit(0)
	}
	choice = strings.TrimSpace(choice)

	switch choice {
	case "1":
		handleNewChat()
	case "2":
		if lastSessionID != "" {
			handleResumeChat()
		} else {
			fmt.Println("No recent session to resume. Please start a new chat.")
		}
	case "3":
		handleHistory()
	case "4":
		authToken = ""
		lastSessionID = ""
		fmt.Println("Logged out")
	case "5":
		fmt.Println("Goodbye!")
		os.Exit(0)
	default:
		fmt.Println("Invalid choice")
	}
}

func prompt(label string) string {
	fmt.Print(label)
	input, err := reader.ReadString('\n')
	if err != nil {
		os.Exit(0)
	}
	return strings.TrimSpace(input)
}

func handleRegister() {
	username := prompt("Username: ")
	email := prompt("Email: ")
	password := prompt("Password: ")

	data := map[string]string{
		"username": username,
		"email":    email,
		"password": password,
	}
	jsonData, _ := json.Marshal(data)

	resp, err := client.Post(baseURL+"/auth/register", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Registration failed: %s\n", string(body))
		return
	}

	fmt.Println("Registration successful! Please login.")
}

func handleLogin() {
	username := prompt("Username: ")
	password := prompt("Password: ")

	data := map[string]string{
		"username": username,
		"password": password,
	}
	jsonData, _ := json.Marshal(data)

	resp, err := client.Post(baseURL+"/auth/login", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Login failed: %s\n", string(body))
		return
	}

	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	if token, ok := result["access_token"].(string); ok && token != "" {
		authToken = token
		fmt.Println("Login successful!")
	} else {
		fmt.Println("Login failed: invalid response or empty token")
	}
}

func handleNewChat() {
	title := prompt("Session Title (optional): ")
	if title == "" {
		title = "New Chat"
	}

	sessionID, err := createSession(title)
	if err != nil {
		fmt.Printf("Failed to create session: %v\n", err)
		return
	}
	fmt.Printf("Session created: %s\n", sessionID)
	lastSessionID = sessionID
	enterChatLoop(sessionID)
}

func handleResumeChat() {
	fmt.Printf("Resuming session: %s\n", lastSessionID)
	enterChatLoop(lastSessionID)
}

func enterChatLoop(sessionID string) {
	fmt.Println("Type 'exit' to quit chat.")
	for {
		msg := prompt("You: ")
		if msg == "exit" {
			break
		}
		if msg == "" {
			continue
		}

		streamChat(sessionID, msg)
	}
}

func createSession(title string) (string, error) {
	data := map[string]string{"title": title}
	jsonData, _ := json.Marshal(data)

	req, _ := http.NewRequest("POST", baseURL+"/chat/sessions", bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.SessionID, nil
}

func streamChat(sessionID, message string) {
	data := map[string]string{
		"session_id": sessionID,
		"message":    message,
		// "model": "gpt-3.5-turbo", // optional
	}
	jsonData, _ := json.Marshal(data)

	req, _ := http.NewRequest("POST", baseURL+"/chat/sessions/stream", bytes.NewBuffer(jsonData))
	req.Header.Set("Authorization", "Bearer "+authToken)
	req.Header.Set("Content-Type", "application/json")
	// Set a longer timeout for streaming
	client := &http.Client{Timeout: 0}

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("Error sending message: %v\n", err)
		return
	}
	defer resp.Body.Close()

	fmt.Print("Bot: ")

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			jsonStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			var event struct {
				Content   string `json:"content"`
				Finished  bool   `json:"finished"`
				SessionID string `json:"sessionId"`
				Message   string `json:"message"` // Error message
			}
			if err := json.Unmarshal([]byte(jsonStr), &event); err == nil {
				if event.Message != "" {
					fmt.Printf("\nError: %s\n", event.Message)
					return
				}
				fmt.Print(event.Content)
				if event.Finished {
					fmt.Println()
					return
				}
			}
		} else if strings.HasPrefix(line, "event: error") {
			// Handle error event
		}
	}
	fmt.Println()
}

func handleHistory() {
	defaultPrompt := "Enter Session ID"
	if lastSessionID != "" {
		defaultPrompt += fmt.Sprintf(" (default: %s)", lastSessionID)
	}
	defaultPrompt += ": "

	sessionID := prompt(defaultPrompt)
	if sessionID == "" && lastSessionID != "" {
		sessionID = lastSessionID
	}

	if sessionID == "" {
		fmt.Println("Session ID is required")
		return
	}

	req, _ := http.NewRequest("GET", fmt.Sprintf("%s/chat/sessions/%s/history", baseURL, sessionID), nil)
	req.Header.Set("Authorization", "Bearer "+authToken)

	resp, err := client.Do(req)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Println("Failed to retrieve history")
		return
	}

	var result struct {
		Messages []struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			Timestamp int64  `json:"timestamp"`
		} `json:"messages"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	for _, msg := range result.Messages {
		fmt.Printf("[%s] %s: %s\n", time.Unix(msg.Timestamp, 0).Format("15:04:05"), msg.Role, msg.Content)
	}
}
