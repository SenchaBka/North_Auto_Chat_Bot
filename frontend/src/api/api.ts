export interface ChatRequest {
  query: string;
  sessionid?: string;
}

export interface ChatResponse {
  answer: string;
  sources: string[];
  toolused?: string;
}

export interface HistoryRequest {
  sessionid: string;
}

export interface HistoryResponse {
  messages: { [key: string]: any }[];
}

const API_BASE_URL = 'http://localhost:8000';

export async function sendChatQuery(query: string, sessionid?: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, sessionid }),
  });

  if (!response.ok) {
    throw new Error('Failed to send chat query');
  }
  return response.json();
}

export async function getChatHistory(sessionid: string): Promise<HistoryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chathistory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionid }),
  });

  if (!response.ok) {
    throw new Error('Failed to fetch chat history');
  }
  return response.json();
}
