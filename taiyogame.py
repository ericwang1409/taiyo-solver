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
space.gravity = (0, 981)  # Negative since y goes down in most renderers

# ball image
radius = 20
ball_image = pygame.image.load('images/sun.png')
ball_image = pygame.transform.scale(ball_image, (radius * 2, radius * 2))  # The image is scaled to 2x the radius of the ball

# Create static lines to form a U-shape
static_lines = [
    pymunk.Segment(space.static_body, (150, 300), (150, 500), 5),
    pymunk.Segment(space.static_body, (150, 500), (450, 500), 5),
    pymunk.Segment(space.static_body, (450, 500), (450, 300), 5)
]
for line in static_lines:
    line.elasticity = 0.95  # To make the ball bounce a bit
    space.add(line)

class Ball:
    def __init__(self, space, position, mass, radius, image_path, scale_factor=1):
        self.radius = radius
        self.is_resting = False
        
        # Pymunk physics setup
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.95
        space.add(self.body, self.shape)
        
        # Load the image for the ball
        self.image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, (int(self.radius * 2 * scale_factor), int(self.radius * 2 * scale_factor)))
        
    def update(self, dt):
        # The physical position will be updated by the Pymunk space.step() method
        pass
    
    def draw(self, screen):
        # Get the position for Pygame (adjust for the coordinate system if needed)
        position = self.body.position.x, self.body.position.y  # No flipping of y-coordinate in this example
        
        # Get a rect structure from the image
        rect = self.image.get_rect(center=position)
        
        # Blit the image onto the screen
        screen.blit(self.image, rect.topleft)

# Create a dynamic ball
# mass = 1
# radius = 25
# moment = pymunk.moment_for_circle(mass, 0, radius)  # 1st argument is mass, 2nd is inner radius, 3rd is outer radius
# ball_body = pymunk.Body(mass, moment)
# ball_body.position = (300, 200)  # Starting position above the U-shape
# ball_shape = pymunk.Circle(ball_body, radius)
# ball_shape.elasticity = 0.95
# space.add(ball_body, ball_shape)

balls = [
    Ball(space, (300, 200), 1, 25, "images/sun.png"),
    Ball(space, (280, 100), 1, 25, "images/venus.png"),
]

# Simulation loop
for i in range(300):
    # Step the simulation
    space.step(1 / 50.0)
    # print(ball_body.position)  #a Optionally print the position of the ball

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))  # Fill the screen with white background
    space.debug_draw(draw_options)  # Draw the space with the debug_draw util

    for ball in balls:
        ball.update(1 / 50.0)

    # Draw the balls
    for ball in balls:
        ball.draw(screen)

    # ball_position = int(ball_body.position.x), int(ball_body.position.y)  # Flip the y-coordinate for Pygame
    # ball_rect = ball_image.get_rect(center=ball_position)
    # screen.blit(ball_image, ball_rect)

    pygame.display.flip()  # Update the full display Surface to the screen

    space.step(1 / 50.0)  # Step the simulation
    clock.tick(50)  # Limit the frame rate to 50 frames per second

pygame.quit()
