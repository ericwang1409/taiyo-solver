import pygame
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
dt = 0

WIDTH = screen.get_width()
HEIGHT = screen.get_height()
BALL_RADIUS=40

# Background image
background_image = pygame.image.load("images/background.jpg").convert()
background_image_rect = background_image.get_rect()
background_image_width = background_image_rect.width
background_image_height = background_image_rect.height
background_image = pygame.transform.scale(background_image, (background_image_width * 0.65, background_image_height * 0.65))

# Barrel box setup
box_height = HEIGHT // 1.4
box_width = box_height // 1.42857142857
box_x = (WIDTH - box_width) // 2  # Horizontally center
box_y = (HEIGHT - box_height) // 2 # Vertically center

bottom_left = (box_x, box_y + box_height)
bottom_right = (box_x + box_width, box_y + box_height)
top_right = (box_x + box_width, box_y)
top_left = (box_x, box_y)

# Score box setup
score_box_width = box_width // 1.5 # Half the width of the box
score_box_height = (HEIGHT - (box_y + box_height)) / 1.5 # Two thirds of the space below the box
score_box_x = (WIDTH - score_box_width) // 2  # Horizontally center
score_box_y = (HEIGHT + (box_y + box_height) - score_box_height) // 2 # Vertically center below box

score_bottom_left = (score_box_x, score_box_y + score_box_height)
score_bottom_right = (score_box_x + score_box_width, score_box_y + score_box_height)
score_top_right = (score_box_x + score_box_width, score_box_y)
score_top_left = (score_box_x, score_box_y)

# Create a space and set the gravity
space = pymunk.Space()
space.gravity = (0, -981)  # Negative since y goes down in most renderers

# Create static lines to form a U-shape
static_lines = [
    pymunk.Segment(space.static_body, bottom_left, top_left, 5),
    pymunk.Segment(space.static_body, bottom_left, bottom_right, 5),
    pymunk.Segment(space.static_body, bottom_right, top_right, 5)
]

for line in static_lines:
    line.elasticity = 0.95  # To make the ball bounce a bit
    space.add(line)

# All ball features
ball_radii = [box_width / 26.96, box_width / 18.78, box_width / 13.19, box_width / 12.125, box_width / 9.46, box_width / 7.76, box_width / 7.53, box_width / 5.64, box_width / 4.67, box_width / 3.29, box_width / 3.54]
scale_factors = [2.4, 2.9, 2.4, 2.4, 2.4, 2.7, 2.9, 3, 2.2, 3, 4]
planet_names=['pluto','moon','mercury','mars','venus','earth','neptune','uranus','saturn','jupiter','sun']

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

# Define balls list
balls = []

ball_dropping = False

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and not ball_dropping:
                ball_dropping = True

    screen.fill((255, 255, 255))  # Fill the screen with white background
    space.debug_draw(draw_options)  # Draw the space with the debug_draw util
    pygame.display.flip()  # Update the full display Surface to the screen

    # Draw scoreboard
    # Position scoreboard boundaries
    # TODO Put an image there
    # Place score centered in box
    pygame.draw.line(screen, "white", score_bottom_left, score_bottom_right, 4)
    pygame.draw.line(screen, "white", score_bottom_left, score_top_left, 4)
    pygame.draw.line(screen, "white", score_bottom_right, score_top_right, 4)
    pygame.draw.line(screen, "white", score_top_left, score_top_right, 4)
    score = 100 # TODO Don't hardcode this
    font_size = 30
    font = pygame.font.Font(None,font_size)
    score_text = font.render(str(score), True, pygame.Color('white'))
    score_rect = score_text.get_rect()

    # Calculate the center position
    score_box_center_x = (score_bottom_left[0] + score_top_right[0]) / 2
    score_box_center_y = (score_bottom_left[1] + score_top_left[1]) / 2
    score_rect.center = (score_box_center_x, score_box_center_y)

    keys = pygame.key.get_pressed()
    mouse = pygame.mouse.get_pos()
    if not ball_dropping:
        if keys[pygame.K_a]:
            current_ball.position.x -= 300 * dt
        if keys[pygame.K_d]:
            current_ball.position.x += 300 * dt
        if (box_x) < mouse[0] < (box_x + box_width):
            current_ball.position.x = mouse[0]

    space.step(1 / 50.0)  # Step the simulation
    clock.tick(50)  # Limit the frame rate to 50 frames per second

pygame.quit()
