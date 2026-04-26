"""
Minimal interactive Overcooked-AI renderer using pygame.

Run from the overcooked_ai project root:
  python testing/pygame_human_play.py --layout cramped_room
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF,
        HWSURFACE,
        QUIT,
        RESIZABLE,
        VIDEORESIZE,
    )
except ModuleNotFoundError as e:  # pragma: no cover
    if e.name != "pygame":
        raise
    print(
        "pygame is not installed. Make sure you're running inside the uv venv.\n\n"
        "From the overcooked_ai project root, run:\n"
        "  uv run python testing/pygame_human_play.py --layout cramped_room\n"
    )
    raise

from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


@dataclass
class ControlScheme:
    up: int
    down: int
    left: int
    right: int
    interact: int
    stay: Optional[int] = None


def _key_to_action(event_key: int, scheme: ControlScheme):
    if event_key == scheme.up:
        return Direction.NORTH
    if event_key == scheme.down:
        return Direction.SOUTH
    if event_key == scheme.left:
        return Direction.WEST
    if event_key == scheme.right:
        return Direction.EAST
    if event_key == scheme.interact:
        return Action.INTERACT
    if scheme.stay is not None and event_key == scheme.stay:
        return Action.STAY
    return None


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="cramped_room", help="layout name")
    p.add_argument(
        "--old-dynamics",
        action="store_true",
        help=(
            "use old cooking dynamics (auto-start cooking when pot becomes full); "
            "by default, cooking starts on INTERACT"
        ),
    )
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tile-size", type=int, default=75)
    p.add_argument(
        "--solo",
        action="store_true",
        help="only control player 0; player 1 always stays",
    )
    p.add_argument(
        "--hide-hud",
        action="store_true",
        help="do not render HUD (slightly faster)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    Recipe.configure({})
    mdp = OvercookedGridworld.from_layout_name(
        args.layout, old_dynamics=bool(args.old_dynamics)
    )
    env = OvercookedEnv.from_mdp(mdp, horizon=args.horizon, info_level=0)

    pygame.init()
    pygame.font.init()
    dyn = "old" if args.old_dynamics else "new"
    pygame.display.set_caption(
        f"Overcooked-AI | layout={args.layout} | dynamics={dyn}"
    )

    # A little repeat helps continuous motion without spamming KEYDOWN logic manually.
    pygame.key.set_repeat(140, 45)

    scheme_p0 = ControlScheme(
        up=pygame.K_UP,
        down=pygame.K_DOWN,
        left=pygame.K_LEFT,
        right=pygame.K_RIGHT,
        interact=pygame.K_SPACE,
        stay=pygame.K_RCTRL,
    )
    scheme_p1 = ControlScheme(
        up=pygame.K_w,
        down=pygame.K_s,
        left=pygame.K_a,
        right=pygame.K_d,
        interact=pygame.K_LSHIFT,
        stay=pygame.K_TAB,  # rarely used; tab is also handled globally
    )

    visualizer = StateVisualizer(
        tile_size=args.tile_size,
        window_fps=args.fps,
        is_rendering_hud=not args.hide_hud,
    )
    clock = pygame.time.Clock()
    ui_font = pygame.font.SysFont(None, 18)

    total_sparse_reward = 0
    total_shaped_reward = 0
    last_joint_action = (Action.STAY, Action.STAY)

    def reset():
        nonlocal total_sparse_reward, total_shaped_reward, last_joint_action
        env.reset(regen_mdp=False)
        total_sparse_reward = 0
        total_shaped_reward = 0
        last_joint_action = (Action.STAY, Action.STAY)

    reset()

    def render_surface() -> pygame.Surface:
        rewards_dict = {
            "score": int(total_sparse_reward),
            "shaped": float(total_shaped_reward),
            "time_left": int(max(0, args.horizon - env.state.timestep)),
            "last_action": Action.joint_action_to_char(last_joint_action),
        }
        hud_data = (
            StateVisualizer.default_hud_data(env.state, **rewards_dict)
            if visualizer.is_rendering_hud
            else None
        )
        return visualizer.render_state(
            state=env.state,
            grid=env.mdp.terrain_mtx,
            hud_data=hud_data,
        )

    base_surface = render_surface()
    window = pygame.display.set_mode(
        base_surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE
    )

    show_help = True
    active_player = 0  # used only when --solo is enabled

    running = True
    while running:
        clock.tick(args.fps)
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                break

            if event.type == VIDEORESIZE:
                window = pygame.display.set_mode(
                    event.dict["size"], HWSURFACE | DOUBLEBUF | RESIZABLE
                )

            if event.type != pygame.KEYDOWN:
                continue

            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
                break

            if event.key == pygame.K_r:
                reset()
                continue

            if event.key == pygame.K_h:
                show_help = not show_help
                continue

            if event.key == pygame.K_TAB:
                active_player = 1 - active_player
                continue

            if env.is_done():
                # Only allow reset/quit/help when episode is done.
                continue

            a0 = _key_to_action(event.key, scheme_p0)
            a1 = None if args.solo else _key_to_action(event.key, scheme_p1)

            if args.solo:
                if active_player == 0:
                    a0 = a0
                    a1 = Action.STAY
                else:
                    a1 = a0
                    a0 = Action.STAY

            if a0 is None and a1 is None:
                continue

            if a0 is None:
                a0 = Action.STAY
            if a1 is None:
                a1 = Action.STAY

            last_joint_action = (a0, a1)
            _next_state, sparse_r, _done, info = env.step(last_joint_action)
            total_sparse_reward += sparse_r
            if info and "shaped_r_by_agent" in info:
                total_shaped_reward += float(sum(info["shaped_r_by_agent"]))

        base_surface = render_surface()

        # Build bottom info bar (so it doesn't cover the game surface).
        footer_lines: list[str] = []
        if show_help:
            footer_lines.append(
                "P0: Arrows move | Space interact | RightCtrl stay  ||  "
                "P1: WASD move | LeftShift interact  ||  "
                "Global: R reset | H toggle bar | ESC/Q quit"
            )
            if args.solo:
                footer_lines.append(
                    f"SOLO: Tab switches controlled player (current: P{active_player})"
                )

        status = (
            f"t={env.state.timestep}/{args.horizon}  "
            f"score={int(total_sparse_reward)}  "
            f"shaped={total_shaped_reward:.2f}  "
            f"last={Action.joint_action_to_char(last_joint_action)}  "
            f"dynamics={'old' if args.old_dynamics else 'new'}"
        )
        if env.is_done():
            status += "  ||  DONE (press R to reset)"
        footer_lines.append(status)

        footer_surface = None
        footer_h = 0
        if footer_lines:
            pad_x, pad_y = 10, 6
            rendered = [ui_font.render(t, True, (255, 255, 255)) for t in footer_lines]
            line_h = max(rs.get_height() for rs in rendered) if rendered else 0
            footer_h = pad_y * 2 + line_h * len(rendered) + 2 * (len(rendered) - 1)
            footer_surface = pygame.Surface((window.get_width(), footer_h), pygame.SRCALPHA)
            footer_surface.fill((0, 0, 0, 190))
            y = pad_y
            for rs in rendered:
                footer_surface.blit(rs, (pad_x, y))
                y += line_h + 2

        # Scale to current window size (minus footer) while preserving aspect.
        win_w, win_h = window.get_size()
        available_h = max(1, win_h - footer_h)
        surf_w, surf_h = base_surface.get_size()
        scale = min(win_w / surf_w, available_h / surf_h)
        draw_size = (max(1, int(surf_w * scale)), max(1, int(surf_h * scale)))
        draw_surface = (
            base_surface
            if draw_size == base_surface.get_size()
            else pygame.transform.smoothscale(base_surface, draw_size)
        )
        x0 = (win_w - draw_surface.get_width()) // 2
        y0 = (available_h - draw_surface.get_height()) // 2

        window.fill((0, 0, 0))
        window.blit(draw_surface, (x0, y0))

        if footer_surface is not None:
            window.blit(footer_surface, (0, win_h - footer_h))

        pygame.display.flip()

    pygame.display.quit()
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


