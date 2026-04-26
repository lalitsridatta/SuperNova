"""
gradio_app.py — Gradio dashboard for Vaccine Cold Chain Last-Mile Delivery
Runs a live episode using the heuristic agent and displays results visually.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from server.environment import VaccineColdChainEnv
from models import VaccineDeliveryAction


def heuristic_policy(obs) -> VaccineDeliveryAction:
    if obs.ice_remaining < 30:
        return VaccineDeliveryAction(action_type="restock_ice", reasoning="Ice critically low.")
    if obs.emergency_stop:
        emergency = next((s for s in obs.stops_remaining if s.stop_id == obs.emergency_stop), None)
        if emergency:
            return VaccineDeliveryAction(action_type="go_to_stop", stop_id=emergency.stop_id,
                                         reasoning=f"Emergency: {emergency.name}")
    if obs.stops_remaining:
        next_stop = min(obs.stops_remaining, key=lambda s: s.priority)
        return VaccineDeliveryAction(action_type="go_to_stop", stop_id=next_stop.stop_id,
                                     reasoning=f"Priority {next_stop.priority}: {next_stop.name}")
    return VaccineDeliveryAction(action_type="wait", reasoning="No stops remaining.")


def run_episode():
    env = VaccineColdChainEnv()
    obs = env.reset()

    log_lines = []
    ice_vals, temp_vals, reward_vals, steps = [], [], [], []
    total_reward = 0.0
    step = 0

    log_lines.append(f"Episode ID: {env.state.episode_id}")
    log_lines.append(f"Total stops: {env.state.total_stops}")
    log_lines.append(f"Emergency stop: {obs.emergency_stop or 'None'}")
    log_lines.append(f"Outside temp: {obs.outside_temperature}°C\n")

    while not obs.done and step < 30:
        action = heuristic_policy(obs)
        ice_vals.append(obs.ice_remaining)
        temp_vals.append(obs.current_temperature)
        steps.append(step)

        stops_left = [s.stop_id for s in obs.stops_remaining]
        line = (f"Step {step+1:02d} | Ice: {obs.ice_remaining:5.1f} | "
                f"Temp: {obs.current_temperature:.1f}°C | "
                f"Action: {action.action_type}"
                + (f" → {action.stop_id}" if action.stop_id else ""))

        obs = env.step(action)
        total_reward += obs.last_reward
        reward_vals.append(total_reward)
        step += 1

        line += f" | Reward: {obs.last_reward:+.3f} | {obs.info}"
        log_lines.append(line)

    log_lines.append(f"\n{'='*50}")
    log_lines.append(f"Result     : {env.state.terminated_reason}")
    log_lines.append(f"Delivered  : {env.state.stops_delivered}/{env.state.total_stops}")
    log_lines.append(f"Restocks   : {env.state.restock_count}")
    log_lines.append(f"Total reward: {env.state.total_reward:.4f}")

    # Build plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(steps, ice_vals, color="steelblue", linewidth=2, marker="o", markersize=4)
    axes[0].axhline(30, color="red", linestyle="--", alpha=0.6, label="Critical (30)")
    axes[0].set_title("Ice Remaining")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Ice Level")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, temp_vals, color="tomato", linewidth=2, marker="o", markersize=4)
    axes[1].axhspan(2, 8, alpha=0.15, color="green", label="Safe zone (2–8°C)")
    axes[1].set_title("Compartment Temperature")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("°C")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, reward_vals, color="seagreen", linewidth=2, marker="o", markersize=4)
    axes[2].set_title("Cumulative Reward")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Reward")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    summary = (f"Delivered {env.state.stops_delivered}/{env.state.total_stops} stops | "
               f"Total reward: {env.state.total_reward:.3f} | "
               f"Result: {env.state.terminated_reason}")

    return "\n".join(log_lines), fig, summary


with gr.Blocks(title="Vaccine Cold Chain — RL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💉 Vaccine Cold Chain Last-Mile Delivery
    ### OpenEnv RL Environment Demo
    An LLM agent manages vaccine delivery under cold chain constraints.
    Vaccines spoil if temperature goes outside **2–8°C**. The agent must plan routes,
    manage ice, and prioritize emergency stops.
    """)

    run_btn = gr.Button("▶ Run Episode", variant="primary", size="lg")

    summary_box = gr.Textbox(label="Episode Summary", interactive=False)
    plot_out = gr.Plot(label="Episode Metrics")
    log_out = gr.Textbox(label="Step-by-Step Log", lines=20, interactive=False)

    run_btn.click(fn=run_episode, outputs=[log_out, plot_out, summary_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
