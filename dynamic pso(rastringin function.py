import numpy as np

# Rastrigin function definition


def rastrigin(x, y, A):
    return A * 2 + (x ** 2 - A * np.cos(2 * np.pi * x)) + (y ** 2 - A * np.cos(2 * np.pi * y))


# Dynamic PSO implementation


def dynamic_pso(dim=2, swarm_size=30, max_iters=100, change_interval=20, c1=1.496, c2=1.496):
    particles = np.random.rand(swarm_size, dim) * 10
    velocities = np.random.rand(swarm_size, dim)
    personal_best_positions = particles.copy()
    personal_best_values = np.array([float('inf')] * swarm_size)
    global_best_position = None
    global_best_value = float('inf')
    A = 10  # Initial Rastrigin function parameter

    for i in range(max_iters):
        # Change Rastrigin function parameters every change_interval iterations
        if i % change_interval == 0:
            A = np.random.uniform(5, 20)  # Change Rastrigin function parameter

        values = np.array([rastrigin(particle[0], particle[1], A)
                          for particle in particles])

        update_personal_best = values < personal_best_values
        personal_best_values[update_personal_best] = values[update_personal_best]
        personal_best_positions[update_personal_best] = particles[update_personal_best]

        min_index = np.argmin(personal_best_values)
        if personal_best_values[min_index] < global_best_value:
            global_best_value = personal_best_values[min_index]
            global_best_position = personal_best_positions[min_index]

        r1, r2 = np.random.rand(
            swarm_size, dim), np.random.rand(swarm_size, dim)
        velocities = c1 * r1 * (personal_best_positions - particles) + \
            c2 * r2 * (global_best_position - particles)
        particles += velocities

    return global_best_position, global_best_value


# Example usage
if __name__ == "__main__":
    # Define the dimensionality of the problem
    dimension = 2

    # Run dynamic PSO to optimize time-varying Rastrigin function
    best_position, best_value = dynamic_pso(
        dim=dimension, swarm_size=30, max_iters=100, change_interval=20)

    # Print results
    print("Best Position: ", best_position)
    print("Best Value: ", best_value)
