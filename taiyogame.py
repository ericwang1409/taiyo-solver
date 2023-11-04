import pygame
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Create a space and set the gravity
space = pymunk.Space()
space.gravity = (0, -981)  # Negative since y goes down in most renderers

# Create static lines to form a U-shape
static_lines = [
    pymunk.Segment(space.static_body, (150, 100), (150, 300), 5),
    pymunk.Segment(space.static_body, (150, 100), (450, 100), 5),
    pymunk.Segment(space.static_body, (450, 300), (450, 100), 5)
]
for line in static_lines:
    line.elasticity = 0.95  # To make the ball bounce a bit
    space.add(line)

# Create a dynamic ball
mass = 1
radius = 25
moment = pymunk.moment_for_circle(mass, 0, radius)  # 1st argument is mass, 2nd is inner radius, 3rd is outer radius
ball_body = pymunk.Body(mass, moment)
ball_body.position = (300, 400)  # Starting position above the U-shape
ball_shape = pymunk.Circle(ball_body, radius)
ball_shape.elasticity = 0.95
space.add(ball_body, ball_shape)

# Simulation loop
for i in range(300):
    # Step the simulation
    space.step(1 / 50.0)
    print(ball_body.position)  # Optionally print the position of the ball

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fill the screen with white background
    space.debug_draw(draw_options)  # Draw the space with the debug_draw util
    pygame.display.flip()  # Update the full display Surface to the screen

    space.step(1 / 50.0)  # Step the simulation
    clock.tick(50)  # Limit the frame rate to 50 frames per second

pygame.quit()
