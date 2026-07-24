import type { ApiError } from "../../types";
import type { GuandanState } from "./types";

async function request<T>(url: string, body: object): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as T & ApiError;
  if (!response.ok) throw new Error(payload.error ?? `请求失败（${response.status}）`);
  return payload;
}

export function createGuandanGame(level: number, aiMode: string): Promise<GuandanState> {
  return request<GuandanState>("/api/guandan/games", {
    humanSeat: 0,
    level,
    aiMode,
  });
}

export function playGuandanCards(
  gameId: string,
  cardIds: number[] | null,
): Promise<GuandanState> {
  return request<GuandanState>(`/api/guandan/games/${gameId}/actions`, { cardIds });
}

