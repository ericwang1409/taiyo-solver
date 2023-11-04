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
space.gravity = (0, 981)  # Negative since y goes down in most renderers

# All ball features
ball_radii = [box_width / 26.96, box_width / 18.78, box_width / 13.19, box_width / 12.125, box_width / 9.46, box_width / 7.76, box_width / 7.53, box_width / 5.64, box_width / 4.67, box_width / 3.29, box_width / 3.54]
scale_factors = [2.4, 2.9, 2.4, 2.4, 2.4, 2.7, 2.9, 3, 2.2, 3, 4]
planet_names=['pluto','moon','mercury','mars','venus','earth','neptune','uranus','saturn','jupiter','sun']

# ball image
radius = 20
ball_image = pygame.image.load('images/sun.png')
ball_image = pygame.transform.scale(ball_image, (radius * 2, radius * 2))  # The image is scaled to 2x the radius of the ball

# Create static lines to form a U-shape
static_lines = [
    pymunk.Segment(space.static_body, bottom_left, top_left, 2),
    pymunk.Segment(space.static_body, bottom_left, bottom_right, 2),
    pymunk.Segment(space.static_body, bottom_right, top_right, 2)
]

for line in static_lines:
    line.elasticity = 0.95  # To make the ball bounce a bit
    space.add(line)

class Ball:
    def __init__(self, position, mass, planetIndex):
        self.radius = ball_radii[planetIndex]
        self.planet = planet_names[planetIndex]
        self.is_resting = False
        
        # Pymunk physics setup
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.95
        space.add(self.body, self.shape)
        
        # Load the image for the ball
        self.image = pygame.image.load("images/" + self.planet + ".png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (int(self.radius * scale_factors[planetIndex]), int(self.radius * scale_factors[planetIndex])))
        
    def update(self, dt):
        # The physical position will be updated by the Pymunk space.step() method
        pass
    
    def draw(self, screen):
        # Get the position for Pygame (adjust for the coordinate system if needed)
        # position = self.body.position.x, self.body.position.y  # No flipping of y-coordinate in this example
        
        # # Get a rect structure from the image
        # rect = self.image.get_rect(center=position)
        
        # # Blit the image onto the screen
        # screen.blit(self.image, rect.topleft)

        size = self.image.get_size()

        circle_surf = pygame.Surface(size, pygame.SRCALPHA)

        # Draw a circle onto this new surface with the same dimensions as the image
        pygame.draw.circle(circle_surf, (255, 255, 255), (size[0]//2, size[1]//2), self.radius)

        # Get a rect structure from the image
        rect = self.image.get_rect(center=(int(self.body.position.x), int(self.body.position.y)))

        # Blit the original image onto the circle_surf using the circle as a mask (only the parts of the image that overlap with the white circle will be blitted)
        circle_surf.blit(self.image, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

        # Blit the new circle_surf onto the screen
        screen.blit(circle_surf, rect.topleft)

balls = [
    Ball(position=(641, 380), mass=1, planetIndex=9),
    Ball(position=(640, 360), mass=1, planetIndex=10),
]

# Simulation loop
for i in range(300):
    # Step the simulation
    space.step(1 / 50.0)
    # print(ball_body.position)  #a Optionally print the position of the ball

# Define balls list

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

    screen.blit(background_image, (0, 0))   # Fill the screen with the background
    space.debug_draw(draw_options)  # Draw the space with the debug_draw util

    for ball in balls:
        ball.update(1 / 50.0)

    # Draw the balls
    for ball in balls:
        ball.draw(screen)

    # ball_position = int(ball_body.position.x), int(ball_body.position.y)  # Flip the y-coordinate for Pygame
    # ball_rect = ball_image.get_rect(center=ball_position)
    # screen.blit(ball_image, ball_rect)

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
    # if not ball_dropping:
    #     if keys[pygame.K_a]:
    #         current_ball.position.x -= 300 * dt
    #     if keys[pygame.K_d]:
    #         current_ball.position.x += 300 * dt
    #     if (box_x) < mouse[0] < (box_x + box_width):
    #         current_ball.position.x = mouse[0]

    pygame.display.flip()  # Update the full display Surface to the screen
    space.step(1 / 50.0)  # Step the simulation
    clock.tick(50)  # Limit the frame rate to 50 frames per second

pygame.quit()
