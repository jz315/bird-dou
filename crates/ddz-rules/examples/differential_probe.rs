use std::io::{self, BufRead, Write};

use ddz_core::{GameAction, GameState, RankCounts, Seat, StepResult};
use ddz_rules::{detect_move_with_rules, PostBidGame, RuleConfig};
use serde::{Deserialize, Serialize};

const DOUZERO_POST_BID_YAML: &str = include_str!("../../../configs/rules/douzero_post_bid.yaml");

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Reset {
        hands: [RankCounts; 3],
        bottom_cards: RankCounts,
        landlord: Seat,
    },
    Step {
        cards: RankCounts,
    },
    Snapshot,
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response {
    Ok { snapshot: Snapshot },
    Error { message: String },
    Bye,
}

#[derive(Debug, Serialize)]
struct Snapshot {
    current_player: Seat,
    hands: [RankCounts; 3],
    played_cards: [RankCounts; 3],
    cards_left: [u8; 3],
    last_non_pass: Option<RankCounts>,
    last_non_pass_player: Option<Seat>,
    consecutive_passes: u8,
    bomb_count: u8,
    multiplier_exp: u8,
    terminal: bool,
    winner: Option<Seat>,
    raw_payoff: [i32; 3],
    objective_payoff: [i32; 3],
    legal_actions: Vec<RankCounts>,
}

struct Session {
    rules: RuleConfig,
    game: Option<PostBidGame>,
    last_result: Option<StepResult>,
}

impl Session {
    fn new() -> Result<Self, String> {
        let rules = RuleConfig::from_yaml_str(DOUZERO_POST_BID_YAML)
            .map_err(|error| format!("invalid checked-in rules: {error}"))?;
        Ok(Self {
            rules,
            game: None,
            last_result: None,
        })
    }

    fn handle(&mut self, request: Request) -> Result<Option<Snapshot>, String> {
        match request {
            Request::Reset {
                hands,
                bottom_cards,
                landlord,
            } => {
                self.game = Some(
                    PostBidGame::new(hands, bottom_cards, landlord, self.rules)
                        .map_err(|error| error.to_string())?,
                );
                self.last_result = None;
                self.snapshot().map(Some)
            }
            Request::Step { cards } => {
                let played = detect_move_with_rules(cards, &self.rules)
                    .map_err(|error| error.to_string())?;
                let game = self
                    .game
                    .as_mut()
                    .ok_or_else(|| "reset must precede step".to_owned())?;
                self.last_result = Some(
                    game.step(&GameAction::Play(played))
                        .map_err(|error| error.to_string())?,
                );
                self.snapshot().map(Some)
            }
            Request::Snapshot => self.snapshot().map(Some),
            Request::Shutdown => Ok(None),
        }
    }

    fn snapshot(&self) -> Result<Snapshot, String> {
        let game = self
            .game
            .as_ref()
            .ok_or_else(|| "reset must precede snapshot".to_owned())?;
        let state = game.state();
        let legal_actions = game
            .legal_moves()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|played| *played.cards())
            .collect();
        let objective_payoff = self
            .last_result
            .as_ref()
            .filter(|result| result.terminal)
            .map_or([0; 3], |result| result.objective_payoff);

        Ok(snapshot_from_state(state, objective_payoff, legal_actions))
    }
}

fn snapshot_from_state(
    state: &GameState,
    objective_payoff: [i32; 3],
    legal_actions: Vec<RankCounts>,
) -> Snapshot {
    Snapshot {
        current_player: state.current_player,
        hands: state.hands,
        played_cards: state.played_cards,
        cards_left: state.cards_left,
        last_non_pass: state.last_non_pass.map(|played| *played.cards()),
        last_non_pass_player: state.last_non_pass_player,
        consecutive_passes: state.consecutive_passes,
        bomb_count: state.bomb_count,
        multiplier_exp: state.multiplier_exp,
        terminal: state.terminal,
        winner: state.terminal.then_some(state.current_player),
        raw_payoff: state.raw_payoff,
        objective_payoff,
        legal_actions,
    }
}

fn main() {
    let mut session = match Session::new() {
        Ok(session) => session,
        Err(message) => {
            let _ = write_response(&Response::Error { message });
            return;
        }
    };
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let response = match line {
            Ok(line) => match serde_json::from_str::<Request>(&line) {
                Ok(Request::Shutdown) => {
                    let _ = write_response(&Response::Bye);
                    break;
                }
                Ok(request) => match session.handle(request) {
                    Ok(Some(snapshot)) => Response::Ok { snapshot },
                    Ok(None) => Response::Bye,
                    Err(message) => Response::Error { message },
                },
                Err(error) => Response::Error {
                    message: format!("invalid request: {error}"),
                },
            },
            Err(error) => Response::Error {
                message: format!("failed to read request: {error}"),
            },
        };
        if write_response(&response).is_err() {
            break;
        }
    }
}

fn write_response(response: &Response) -> io::Result<()> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, response)?;
    writeln!(output)?;
    output.flush()
}
