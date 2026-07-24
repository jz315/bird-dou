import { lazy, Suspense, useState } from "react";

const DdzGame = lazy(() => import("./modes/ddz/DdzGame"));
const GuandanGame = lazy(() => import("./modes/guandan/GuandanGame"));

type GameMode = "ddz" | "guandan";

function initialMode(): GameMode {
  return window.location.hash === "#guandan" ? "guandan" : "ddz";
}

function App() {
  const [mode, setMode] = useState<GameMode>(initialMode);

  const switchMode = (next: GameMode) => {
    window.location.hash = next === "guandan" ? "guandan" : "ddz";
    setMode(next);
  };

  return (
    <>
      <nav className="game-mode-switch" aria-label="选择玩法">
        <button className={mode === "ddz" ? "active" : ""} onClick={() => switchMode("ddz")}>
          斗地主
        </button>
        <button
          className={mode === "guandan" ? "active" : ""}
          onClick={() => switchMode("guandan")}
        >
          两副牌掼蛋
        </button>
      </nav>
      <Suspense fallback={<div className="mode-loading">正在铺牌桌…</div>}>
        {mode === "ddz" ? <DdzGame /> : <GuandanGame />}
      </Suspense>
    </>
  );
}

export default App;
