use std::collections::BTreeSet;

use crate::game::round::{Round, RoundOutcome, TrickTarget};
use crate::game::{Action, GameError, StepResult};
use crate::{beats, detect_move, Card, Move, Seat, PLAYER_COUNT};

impl Round {
    pub(super) fn apply(&mut self, actor: Seat, action: Action) -> Result<StepResult, GameError> {
        if self.outcome.is_some() {
            return Err(GameError::RoundComplete);
        }
        if actor != self.current_player {
            return Err(GameError::NotCurrentPlayer {
                expected: self.current_player,
                actual: actor,
            });
        }

        match action {
            Action::Pass => self.apply_pass(actor),
            Action::Play(cards) => self.apply_play(actor, &cards),
        }
    }

    fn apply_pass(&mut self, actor: Seat) -> Result<StepResult, GameError> {
        if self.target.is_none() {
            return Err(GameError::MustLead);
        }
        self.passes_since_play += 1;
        let trick_ended = usize::from(self.passes_since_play) >= self.required_passes();
        if trick_ended {
            self.current_player = self.next_trick_leader();
            self.target = None;
            self.passes_since_play = 0;
        } else {
            self.current_player = self
                .next_active_after(actor)
                .expect("an unfinished round has another active player");
        }
        Ok(self.result(None, trick_ended))
    }

    fn apply_play(&mut self, actor: Seat, cards: &[Card]) -> Result<StepResult, GameError> {
        for card in cards {
            if !self.hands[actor.index()].contains(*card) {
                return Err(GameError::CardNotOwned(*card));
            }
        }
        let movement = detect_move(cards, self.level)?;
        if self
            .target
            .as_ref()
            .is_some_and(|target| !beats(&movement, &target.movement, self.level))
        {
            return Err(GameError::DoesNotBeat);
        }

        self.hands[actor.index()].remove_all(cards);
        self.target = Some(TrickTarget {
            player: actor,
            movement: movement.clone(),
        });
        self.passes_since_play = 0;
        if self.hands[actor.index()].is_empty() {
            self.finish_order.push(actor);
        }
        self.complete_if_three_finished();
        if self.outcome.is_none() {
            self.current_player = self
                .next_active_after(actor)
                .expect("an unfinished round has another active player");
        }
        Ok(self.result(Some(movement), false))
    }

    fn required_passes(&self) -> usize {
        let active = self.hands.iter().filter(|hand| !hand.is_empty()).count();
        let target_is_active = self
            .target
            .as_ref()
            .is_some_and(|target| !self.hands[target.player.index()].is_empty());
        active - usize::from(target_is_active)
    }

    fn next_trick_leader(&self) -> Seat {
        let target_player = self.target.as_ref().expect("a pass has a target").player;
        if self.hands[target_player.index()].is_empty() {
            let partner = target_player.partner();
            if self.hands[partner.index()].is_empty() {
                self.next_active_after(target_player)
                    .expect("an unfinished round has an active player")
            } else {
                partner
            }
        } else {
            target_player
        }
    }

    fn next_active_after(&self, seat: Seat) -> Option<Seat> {
        let mut candidate = seat.next();
        for _ in 0..PLAYER_COUNT {
            if !self.hands[candidate.index()].is_empty() {
                return Some(candidate);
            }
            candidate = candidate.next();
        }
        None
    }

    fn complete_if_three_finished(&mut self) {
        if self.finish_order.len() != PLAYER_COUNT - 1 {
            return;
        }
        let finished: BTreeSet<_> = self.finish_order.iter().copied().collect();
        let last = Seat::ALL
            .into_iter()
            .find(|seat| !finished.contains(seat))
            .expect("exactly one seat remains");
        self.finish_order.push(last);
        let finish_order: [Seat; PLAYER_COUNT] = self
            .finish_order
            .clone()
            .try_into()
            .expect("four finishing seats were recorded");
        self.outcome = Some(
            RoundOutcome::from_finish_order(finish_order)
                .expect("the state machine records every seat exactly once"),
        );
    }

    fn result(&self, played: Option<Move>, trick_ended: bool) -> StepResult {
        StepResult {
            played,
            trick_ended,
            next_player: self.outcome.is_none().then_some(self.current_player),
            round_outcome: self.outcome.clone(),
        }
    }
}
