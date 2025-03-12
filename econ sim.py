"""
Complex Economic System Simulation

This file simulates a complex economic system with multiple agents, market dynamics, and feedback loops.
The simulation includes various agent strategies, network effects, and emergent behaviors.

Dependencies:
- networkx
- matplotlib
- numpy
- pandas

Note:
This is a simplified model and does not capture all the complexities of real-world economic systems.
The accuracy of the simulation depends on the assumptions made and the specific parameters used.
For more accurate and detailed economic modeling, additional factors such as macroeconomic indicators,
policy interventions, and more sophisticated agent behaviors would need to be considered.
"""

import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation


class EconomicAgent:
    """
    Represents an economic agent (individual, firm, etc.) in the complex economic system
    """

    def __init__(self, agent_id, agent_type, initial_wealth=100, strategy="random"):
        self.id = agent_id
        self.type = agent_type  # consumer, producer, investor, etc.
        self.wealth = initial_wealth
        self.strategy = strategy
        self.connections = []
        self.history = [initial_wealth]
        self.price_expectation = random.uniform(0.8, 1.2)

    def update_price_expectation(self, market_price, learning_rate=0.1):
        """Update price expectations based on market observations"""
        error = market_price - self.price_expectation
        self.price_expectation += learning_rate * error

    def decide_action(self, market_price):
        """Determine buy/sell action based on price expectations"""
        if self.strategy == "random":
            return random.choice(["buy", "sell", "hold"])
        elif self.strategy == "trend_follower":
            if market_price > self.price_expectation:
                return "buy"  # Buy if price is rising
            else:
                return "sell"  # Sell if price is falling
        elif self.strategy == "contrarian":
            if market_price > self.price_expectation:
                return "sell"  # Sell if price seems high
            else:
                return "buy"  # Buy if price seems low
        return "hold"

    def update_wealth(self, amount):
        """Update agent's wealth"""
        self.wealth += amount
        self.history.append(self.wealth)


class ComplexEconomicSystem:
    """
    Represents a complex economic system with agents, market dynamics, and feedback loops
    """

    def __init__(self, num_agents=50):
        self.agents = []
        self.market_price = 100
        self.price_history = [self.market_price]
        self.time_step = 0
        self.network = nx.Graph()

        # Create agents with different strategies
        strategies = ["random", "trend_follower", "contrarian"]
        agent_types = ["consumer", "producer", "investor"]

        for i in range(num_agents):
            agent_type = random.choice(agent_types) # randomly assign agent type (discrete uniform distribution)
            strategy = random.choice(strategies) # randomly assign agent type (discrete uniform distribution)
            initial_wealth = random.uniform(50, 150)
            agent = EconomicAgent(i, agent_type, initial_wealth, strategy)
            self.agents.append(agent)
            self.network.add_node(i, type=agent_type, strategy=strategy, wealth=initial_wealth)

        # Create scale-free network connections (power law distribution)
        # This mimics real economic networks where some entities have many connections
        self._create_scale_free_network()

    def _create_scale_free_network(self):
        """Create a scale-free network using preferential attachment"""
        # Start with a small clique of 3 nodes
        for i in range(min(3, len(self.agents))):
            for j in range(i + 1, min(3, len(self.agents))):
                self.network.add_edge(i, j)
                self.agents[i].connections.append(j)
                self.agents[j].connections.append(i)

        # Add remaining nodes with preferential attachment
        for i in range(3, len(self.agents)):
            # Number of connections for this new node
            num_connections = random.randint(1, 3)

            # Nodes with more connections are more likely to get new connections
            connection_weights = [len(self.agents[j].connections) + 1 for j in range(i)]
            total_weight = sum(connection_weights)
            probabilities = [w / total_weight for w in connection_weights]

            # Create new connections
            new_connections = np.random.choice(
                range(i), size=min(num_connections, i),
                replace=False, p=probabilities
            )

            for j in new_connections:
                self.network.add_edge(i, j)
                self.agents[i].connections.append(j)
                self.agents[j].connections.append(i)

    def update_market_price(self, buys, sells):
        """Update market price based on supply and demand"""
        # Simple price update based on difference between buys and sells
        # Demonstrates negative feedback loop in supply and demand
        price_change = (buys - sells) * 0.1
        self.market_price = max(1, self.market_price + price_change)
        self.price_history.append(self.market_price)

    def run_simulation_step(self):
        """Run a single step of the simulation"""
        self.time_step += 1
        buys = 0
        sells = 0

        # Each agent makes a decision based on their strategy
        for agent in self.agents:
            # Update price expectation based partly on connected agents (network effect)
            connected_expectations = [self.agents[conn].price_expectation
                                      for conn in agent.connections]
            if connected_expectations:
                network_effect = sum(connected_expectations) / len(connected_expectations)
                agent.price_expectation = (0.7 * agent.price_expectation +
                                           0.3 * network_effect)

            action = agent.decide_action(self.market_price)

            if action == "buy":
                buys += 1
                # Pay the market price to buy
                agent.update_wealth(-self.market_price * 0.01)
            elif action == "sell":
                sells += 1
                # Gain the market price from selling
                agent.update_wealth(self.market_price * 0.01)

        # Update market price based on supply and demand
        self.update_market_price(buys, sells)

        # Introduce occasional shocks to demonstrate non-linearity
        if random.random() < 0.05:  # 5% chance of shock
            shock_size = random.uniform(-10, 10)
            self.market_price += shock_size
            print(f"Market shock at time {self.time_step}: {shock_size:.2f}")

        # Return market price and buys/sells ratio for this step
        return self.market_price, buys, sells

    def run_simulation(self, steps=100):
        """Run the simulation for multiple steps"""
        results = []
        for _ in range(steps):
            price, buys, sells = self.run_simulation_step()
            results.append({
                'time': self.time_step,
                'price': price,
                'buys': buys,
                'sells': sells,
                'buy_sell_ratio': buys / max(1, sells)
            })

        return pd.DataFrame(results)

    def get_agent_wealth_history(self):
        """Get the wealth history of all agents"""
        data = {}
        for agent in self.agents:
            data[f"Agent {agent.id} ({agent.type}, {agent.strategy})"] = agent.history

        # Pad shorter histories with None for consistent dataframe creation
        max_len = max(len(history) for history in data.values())
        for key, history in data.items():
            if len(history) < max_len:
                data[key] = history + [None] * (max_len - len(history))

        return pd.DataFrame(data)

    def plot_network(self, highlight_attribute='type'):
        """
        Plot the network of economic agents

        Parameters:
        highlight_attribute: 'type' or 'strategy' to color nodes
        """
        plt.figure(figsize=(10, 8))

        # Position nodes using force-directed layout
        pos = nx.spring_layout(self.network, seed=42)

        # Color nodes based on attribute
        if highlight_attribute == 'type':
            color_map = {'consumer': 'blue', 'producer': 'green', 'investor': 'red'}
            node_colors = [color_map[self.network.nodes[n]['type']] for n in self.network.nodes()]
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10, label=agent_type)
                               for agent_type, color in color_map.items()]
        else:  # strategy
            color_map = {'random': 'purple', 'trend_follower': 'orange', 'contrarian': 'cyan'}
            node_colors = [color_map[self.network.nodes[n]['strategy']] for n in self.network.nodes()]
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10, label=strategy)
                               for strategy, color in color_map.items()]

        # Node size based on wealth
        node_sizes = [self.network.nodes[n]['wealth'] / 2 for n in self.network.nodes()]

        # Draw the network
        nx.draw_networkx(
            self.network, pos,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
            alpha=0.8,
            edge_color='gray',
            width=0.5
        )

        plt.legend(handles=legend_elements)
        plt.title(f"Economic Network - Colored by {highlight_attribute.capitalize()}")
        plt.axis('off')
        plt.tight_layout()
        return plt.gcf()

    def visualize_price_history(self):
        """Visualize the market price history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.price_history, 'b-', linewidth=2)
        plt.title("Market Price Evolution")
        plt.xlabel("Time Step")
        plt.ylabel("Market Price")
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    def analyze_wealth_distribution(self):
        """Analyze and visualize the wealth distribution of agents"""
        plt.figure(figsize=(10, 6))

        final_wealth = [agent.wealth for agent in self.agents]
        plt.hist(final_wealth, bins=15, alpha=0.7, color='green')
        plt.axvline(np.mean(final_wealth), color='red', linestyle='dashed', linewidth=2,
                    label=f'Mean: {np.mean(final_wealth):.2f}')
        plt.axvline(np.median(final_wealth), color='blue', linestyle='dashed', linewidth=2,
                    label=f'Median: {np.median(final_wealth):.2f}')

        plt.title("Wealth Distribution After Simulation")
        plt.xlabel("Wealth")
        plt.ylabel("Number of Agents")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    def animate_wealth_over_time(self, sample_size=10):
        """Create an animation of wealth changes over time for a sample of agents"""
        # Select a subset of agents to visualize
        sampled_agents = random.sample(self.agents, min(sample_size, len(self.agents)))

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        lines = []
        for agent in sampled_agents:
            line, = ax.plot([], [], label=f"Agent {agent.id} ({agent.strategy})")
            lines.append(line)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Wealth')
        ax.set_title('Wealth Evolution Over Time')
        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, alpha=0.3)

        # Set initial x and y limits
        ax.set_xlim(0, len(sampled_agents[0].history))
        min_wealth = min([min(agent.history) for agent in sampled_agents])
        max_wealth = max([max(agent.history) for agent in sampled_agents])
        margin = (max_wealth - min_wealth) * 0.1
        ax.set_ylim(min_wealth - margin, max_wealth + margin)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            for j, agent in enumerate(sampled_agents):
                x = list(range(i + 1))
                y = agent.history[:i + 1]
                lines[j].set_data(x, y)
            return lines

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(sampled_agents[0].history),
                                       interval=200, blit=True)

        return anim

    def create_heatmap_analysis(self):
        """Create a heatmap showing relationships between variables"""
        data = {
            'Time': list(range(len(self.price_history))),
            'Price': self.price_history
        }

        # Extract strategy counts for each time step
        strategy_counts = []
        for t in range(len(self.price_history)):
            if t == 0:
                # Initial distribution
                counts = {strategy: sum(1 for agent in self.agents if agent.strategy == strategy)
                          for strategy in ['random', 'trend_follower', 'contrarian']}
            else:
                # In a real implementation, this would track how strategies changed over time
                # For simplicity, we'll use the initial distribution plus some randomness
                base_counts = {strategy: sum(1 for agent in self.agents if agent.strategy == strategy)
                               for strategy in ['random', 'trend_follower', 'contrarian']}
                counts = {s: max(0, c + random.randint(-2, 2)) for s, c in base_counts.items()}

            strategy_counts.append(counts)

        # Add strategy counts to data
        for strategy in ['random', 'trend_follower', 'contrarian']:
            data[f'{strategy}_count'] = [counts[strategy] for counts in strategy_counts]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Calculate correlations
        correlation_matrix = df.corr()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        cmap = LinearSegmentedColormap.from_list('rg', ["red", "white", "green"], N=256)
        plt.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')

        # Add labels
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

        # Add correlation values
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha='center', va='center', color='black')

        plt.title('Correlation Heatmap Between Economic Variables')
        plt.tight_layout()
        return plt.gcf()


def analyze_path_dependence(num_simulations=5, steps=100, initial_price_range=(80, 120)):
    """
    Analyze path dependence by running multiple simulations with different initial conditions
    """
    plt.figure(figsize=(12, 8))

    for i in range(num_simulations):
        # Create a new system with different initial price
        system = ComplexEconomicSystem(num_agents=50)
        system.market_price = random.uniform(*initial_price_range)
        system.price_history = [system.market_price]

        # Run simulation
        results = system.run_simulation(steps)

        # Plot price history
        plt.plot(results['time'], results['price'],
                 label=f"Sim {i + 1}: Initial Price {system.price_history[0]:.2f}")

    plt.title("Path Dependence in Economic Systems")
    plt.xlabel("Time Steps")
    plt.ylabel("Market Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt.gcf()


def demonstrate_feedback_loops():
    """
    Demonstrate positive and negative feedback loops in economic systems
    """
    # Create a system with specific parameters to demonstrate feedback loops
    system = ComplexEconomicSystem(num_agents=50)

    # Run simulation
    results = system.run_simulation(steps=100)

    # Calculate metrics to identify feedback loops
    results['price_change'] = results['price'].diff()
    results['acceleration'] = results['price_change'].diff()

    # Identify positive feedback periods (accelerating price changes)
    positive_feedback = results[results['acceleration'].abs() > 0.5]

    # Identify negative feedback periods (decelerating price changes)
    negative_feedback = results[(results['acceleration'].abs() <= 0.5) &
                                (results['price_change'].abs() > 0.1)]

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot price
    ax1.plot(results['time'], results['price'], 'b-', linewidth=2)
    ax1.set_ylabel('Market Price')
    ax1.set_title('Market Price Evolution with Feedback Loops')
    ax1.grid(True, alpha=0.3)

    # Highlight positive feedback loops
    for idx in positive_feedback.index:
        if idx > 0:
            ax1.axvspan(results.loc[idx - 1, 'time'], results.loc[idx, 'time'],
                        alpha=0.3, color='red')

    # Highlight negative feedback loops
    for idx in negative_feedback.index:
        if idx > 0:
            ax1.axvspan(results.loc[idx - 1, 'time'], results.loc[idx, 'time'],
                        alpha=0.3, color='green')

    # Plot buy/sell ratio
    ax2.plot(results['time'], results['buy_sell_ratio'], 'g-', linewidth=2)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Buy/Sell Ratio')
    ax2.set_title('Buy/Sell Ratio (Indicator of Market Sentiment)')
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.3, label='Positive Feedback'),
        Patch(facecolor='green', alpha=0.3, label='Negative Feedback')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def analyze_emergent_behaviors(steps=100):
    """
    Analyze emergent behaviors in complex economic systems
    """
    # Create a system
    system = ComplexEconomicSystem(num_agents=50)

    # Run simulation
    results = system.run_simulation(steps)

    # Identify clusters and patterns
    plt.figure(figsize=(12, 10))

    # Plot 1: Price and Volume
    plt.subplot(2, 2, 1)
    plt.plot(results['time'], results['price'], 'b-', label='Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Market Price')
    plt.title('Market Price Evolution')
    plt.grid(True, alpha=0.3)

    ax2 = plt.twinx()
    volume = results['buys'] + results['sells']
    ax2.plot(results['time'], volume, 'r--', label='Volume')
    ax2.set_ylabel('Trading Volume')

    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Plot 2: Wealth Distribution Over Time
    plt.subplot(2, 2, 2)
    wealth_history = system.get_agent_wealth_history()

    # Sample a few agents
    sample_agents = random.sample(wealth_history.columns.tolist(),
                                  min(5, len(wealth_history.columns)))

    for agent in sample_agents:
        plt.plot(wealth_history.index, wealth_history[agent], label=agent)

    plt.xlabel('Time Steps')
    plt.ylabel('Wealth')
    plt.title('Wealth Evolution of Sample Agents')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize='small')

    # Plot 3: State Space Analysis
    plt.subplot(2, 2, 3)
    if len(results) > 1:
        plt.scatter(results['price'][:-1], results['price'][1:],
                    c=results['time'][1:], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Time Step')
        plt.xlabel('Price (t)')
        plt.ylabel('Price (t+1)')
        plt.title('State Space Analysis')
        plt.grid(True, alpha=0.3)

    # Plot 4: Final Wealth Distribution by Strategy
    plt.subplot(2, 2, 4)

    # Get wealth by strategy
    wealth_by_strategy = {strategy: [] for strategy in ['random', 'trend_follower', 'contrarian']}
    for agent in system.agents:
        wealth_by_strategy[agent.strategy].append(agent.wealth)

    # Plot boxplots
    plt.boxplot([wealth_by_strategy[s] for s in ['random', 'trend_follower', 'contrarian']],
                labels=['Random', 'Trend Follower', 'Contrarian'])
    plt.ylabel('Final Wealth')
    plt.title('Wealth Distribution by Strategy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt.gcf()


def compare_strategies(num_simulations=10, steps=100):
    """
    Compare different strategies across multiple simulations
    """
    # Track performance across simulations
    strategy_performance = {
        'random': [],
        'trend_follower': [],
        'contrarian': []
    }

    for _ in range(num_simulations):
        system = ComplexEconomicSystem(num_agents=60)  # 20 of each strategy
        system.run_simulation(steps)

        # Calculate average wealth by strategy
        for strategy in strategy_performance:
            avg_wealth = np.mean([agent.wealth for agent in system.agents
                                  if agent.strategy == strategy])
            strategy_performance[strategy].append(avg_wealth)

    # Plot results
    plt.figure(figsize=(10, 6))

    # Boxplot of strategy performance
    plt.boxplot([strategy_performance[s] for s in ['random', 'trend_follower', 'contrarian']],
                labels=['Random', 'Trend Follower', 'Contrarian'])

    plt.title('Strategy Performance Across Multiple Simulations')
    plt.ylabel('Average Final Wealth')
    plt.grid(True, alpha=0.3)

    # Add individual points
    for i, strategy in enumerate(['random', 'trend_follower', 'contrarian']):
        x = np.random.normal(i + 1, 0.05, size=len(strategy_performance[strategy]))
        plt.scatter(x, strategy_performance[strategy], alpha=0.5)

    plt.tight_layout()
    return plt.gcf()


def main():
    """
    Main function to demonstrate complex economic systems
    """
    # Create a complex economic system
    system = ComplexEconomicSystem(num_agents=50)

    # Run simulation
    print("Running simulation...")
    results = system.run_simulation(steps=100)

    # Create visualizations
    print("Creating visualizations...")

    # 1. Network visualization
    print("1. Network visualization")
    network_fig = system.plot_network(highlight_attribute='type')
    network_fig.savefig('economic_network.png')

    network_fig2 = system.plot_network(highlight_attribute='strategy')
    network_fig2.savefig('economic_network_by_strategy.png')

    # 2. Price evolution
    print("2. Price evolution")
    price_fig = system.visualize_price_history()
    price_fig.savefig('price_evolution.png')

    # 3. Wealth distribution
    print("3. Wealth distribution")
    wealth_fig = system.analyze_wealth_distribution()
    wealth_fig.savefig('wealth_distribution.png')

    # 4. Path dependence
    print("4. Path dependence")
    path_fig = analyze_path_dependence(num_simulations=5, steps=100)
    path_fig.savefig('path_dependence.png')

    # 5. Feedback loops
    print("5. Feedback loops")
    feedback_fig = demonstrate_feedback_loops()
    feedback_fig.savefig('feedback_loops.png')

    # 6. Emergent behaviors
    print("6. Emergent behaviors")
    emergent_fig = analyze_emergent_behaviors(steps=100)
    emergent_fig.savefig('emergent_behaviors.png')

    # 7. Strategy comparison
    print("7. Strategy comparison")
    strategy_fig = compare_strategies()
    strategy_fig.savefig('strategy_comparison.png')

    print("Simulation and analysis complete!")
    print("Files saved as PNG images in the current directory.")

    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"Final market price: {system.market_price:.2f}")
    print(f"Average agent wealth: {np.mean([agent.wealth for agent in system.agents]):.2f}")
    print(f"Wealth inequality (Gini coefficient): {calculate_gini([agent.wealth for agent in system.agents]):.4f}")

    # Strategy performance
    print("\nStrategy Performance:")
    for strategy in ['random', 'trend_follower', 'contrarian']:
        avg_wealth = np.mean([agent.wealth for agent in system.agents if agent.strategy == strategy])
        print(f"Average wealth for {strategy} strategy: {avg_wealth:.2f}")


def calculate_gini(wealth_values):
    """
    Calculate the Gini coefficient as a measure of wealth inequality
    """
    sorted_wealth = np.sort(wealth_values)
    n = len(sorted_wealth)
    cumulative_wealth = np.cumsum(sorted_wealth)
    return (n + 1 - 2 * np.sum(cumulative_wealth) / (cumulative_wealth[-1] * n)) / n


if __name__ == "__main__":
    main()
