import type { ApiError, GameState } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  const payload = (await response.json()) as T & ApiError;
  if (!response.ok) {
    throw new Error(payload.error ?? `请求失败（${response.status}）`);
  }
  return payload;
}

export function createGame(aiMode: string): Promise<GameState> {
  return request<GameState>("/api/games", {
    method: "POST",
    body: JSON.stringify({ humanSeat: 0, aiMode }),
  });
}

export function playAction(gameId: string, actionIndex: number): Promise<GameState> {
  return request<GameState>(`/api/games/${gameId}/actions`, {
    method: "POST",
    body: JSON.stringify({ actionIndex }),
  });
}
