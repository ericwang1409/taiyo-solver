import pygame
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
import random
import time

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
draw_options = pymunk.pygame_util.DrawOptions(screen)
dt = 0

WIDTH = screen.get_width()
HEIGHT = screen.get_height()
BALL_RADIUS = 40

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

# side image
side_image = pygame.image.load('images/wheel.png').convert_alpha()
side_image_rect = side_image.get_rect()
side_image_position = (top_left[0] - 350, top_left[1])
side_image = pygame.transform.scale(side_image, (side_image_rect.width // 4, side_image_rect.height // 4))

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
    pymunk.Segment(space.static_body, bottom_left, top_left, 3),
    pymunk.Segment(space.static_body, bottom_left, bottom_right, 5),
    pymunk.Segment(space.static_body, bottom_right, top_right, 3)
]

for line in static_lines:
    line.elasticity = 0  # To make the ball bounce a bit
    line.friction = 0.6
    space.add(line)

class Ball:
    def __init__(self, position, mass, planetIndex, bodytype=pymunk.Body.KINEMATIC):
        self.radius = ball_radii[planetIndex]
        self.planet = planet_names[planetIndex]
        self.planetIndex = planetIndex
        self.is_resting = False
        self.mass = mass
        
        # Pymunk physics setup
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.moment = moment
        self.body = pymunk.Body(mass, moment, bodytype)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0
        self.shape.friction = 0.5
        self.shape.collision_type = 1  # Assign a collision type for balls
        space.add(self.body, self.shape) # TODO Add collision handler
        
        # Load the image for the ball
        self.image = pygame.image.load("images/" + self.planet + ".png").convert_alpha()
        self.image = pygame.transform.scale(self.image, (int(self.radius * scale_factors[planetIndex]), int(self.radius * scale_factors[planetIndex])))
        self.shape.body.data = self
        
    def update(self, dt):
        # The physical position will be updated by the Pymunk space.step() method
        ball.body.angular_velocity *= 0.9
    
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

    def delete(self, space):
        # Remove the shape and body from the space
        space.remove(self.shape, self.body)
        if self in balls: # Might not want to reference balls here? Though it is a global var I think.
            balls.remove(self)

# Define balls list
balls = []

# Returns the position of a new ball
def spawn_new_ball(planet_index):
    return Ball(position=(640,50), mass=1, planetIndex=planet_index)          #Ball(position=(WIDTH / 2, box_y - ball_radii[planet_index]), mass=1, planetIndex=planet_index)

current_ball = spawn_new_ball(0)

score = 0

# Game over screen function
def show_game_over_screen():
    screen.fill((0, 0, 0))  # Fill the screen with black or any other color for the game over screen

    # Display the score
    font = pygame.font.Font(None, 74)
    text = font.render(f'Game Over! Your Score: {score}', True, (255, 255, 255))
    text_rect = text.get_rect(center=(WIDTH / 2, HEIGHT / 3))
    screen.blit(text, text_rect)

    # Draw the replay button
    button_color = (0, 255, 0)  # Green color
    button_rect = pygame.Rect(WIDTH / 2 - 100, HEIGHT / 2, 200, 60)  # Create a rect for the button
    pygame.draw.rect(screen, button_color, button_rect)  # Draw the button

    # Button text
    button_text = font.render('Replay', True, (0, 0, 0))
    button_text_rect = button_text.get_rect(center=button_rect.center)
    screen.blit(button_text, button_text_rect)

    pygame.display.flip()  # Update the display

    # Wait for the player to click the replay button
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = event.pos
                if button_rect.collidepoint(mouse_pos):
                    waiting = False

# State clear (for starting new game instance)
def clear_balls(space, balls, current_ball):
    if current_ball in balls:
        balls.remove(current_ball)
    if current_ball.shape in space.shapes:
        space.remove(current_ball.shape, current_ball.body)

    for ball in balls:
        # Remove the shape and body from the space
        space.remove(ball.shape, ball.body)
    balls.clear() 

# Set up collision handler and collision_callback function
handler = space.add_collision_handler(1, 1)

def ball_collision_handler(arbiter, space, data):
    global score
    ball_shape1, ball_shape2 = arbiter.shapes

    # Check if both shapes are balls and have the same radius
    if ball_shape1.radius == ball_shape2.radius:

        ball1 = ball_shape1.body.data
        ball2 = ball_shape2.body.data

        # Determine the lower ball's position
        planetIndex = ball1.planetIndex
        if (planetIndex < len(planet_names)-1) and (ball1.body.body_type == ball2.body.body_type):

            lower_ball = ball1 if ball1.body.position[1] < ball2.body.position[1] else ball2
            new_position = lower_ball.body.position

            # Create a new Ball instance at the position of the lower ball
            new_ball = Ball(new_position, 10, planetIndex+1, bodytype=pymunk.Body.DYNAMIC)
            ball1.delete(space)
            ball2.delete(space)
            balls.append(new_ball)

            score += 2*(planetIndex+1)

    return True
handler.begin = ball_collision_handler

ball_dropping = False

# Frame counter for drop timing
frame_count = 0
current_frames = 90

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1) or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
            if not ball_dropping and current_ball.body.body_type == pymunk.Body.KINEMATIC:
                ball_dropping = True
                current_ball.body.body_type = pymunk.Body.DYNAMIC
                current_ball.body.mass = 1
                current_ball.body.moment = pymunk.moment_for_circle(1, 0, current_ball.radius, (0, 0))
                space.reindex_shapes_for_body(current_ball.body)
                balls.append(current_ball)
                if frame_count == 0:
                    frame_count = 1
                    current_frames = 1
                else:
                    current_frames = frame_count

    screen.blit(background_image, (0, 0))   # Fill the screen with the background
    screen.blit(side_image, side_image_position)  # Draw the side image
    space.debug_draw(draw_options)  # Draw the space with the debug_draw util
       
    current_ball.draw(screen)
    # print(current_ball.body.position)
    # Draw the balls
    for ball in balls:
        ball.draw(screen)
        ball.update(1 / 50.0)

    # Draw scoreboard
    # Position scoreboard boundaries
    # TODO Put an image there
    # Place score centered in box
    pygame.draw.line(screen, "white", score_bottom_left, score_bottom_right, 4)
    pygame.draw.line(screen, "white", score_bottom_left, score_top_left, 4)
    pygame.draw.line(screen, "white", score_bottom_right, score_top_right, 4)
    pygame.draw.line(screen, "white", score_top_left, score_top_right, 4)
    font_size = 30
    font = pygame.font.Font(None,font_size)
    score_text = font.render(str(score), True, pygame.Color('white'))
    score_rect = score_text.get_rect()

    # Calculate the center position
    score_box_center_x = (score_bottom_left[0] + score_top_right[0]) / 2
    score_box_center_y = (score_bottom_left[1] + score_top_left[1]) / 2
    score_rect.center = (score_box_center_x, score_box_center_y)

    # Draw the score text onto the screen
    screen.blit(score_text, score_rect.topleft)

    # Movement for the dropper-position ball
    keys = pygame.key.get_pressed()
    mouse = pygame.mouse.get_pos()

    if not ball_dropping and current_ball.body.body_type == pymunk.Body.KINEMATIC:
        if keys[pygame.K_a]:
            current_ball.body.position = pymunk.Vec2d(current_ball.body.position.x - (300 * dt), current_ball.body.position.y)
        if keys[pygame.K_d]:
            current_ball.body.position = pymunk.Vec2d(current_ball.body.position.x + (300 * dt), current_ball.body.position.y)
        if (box_x) < mouse[0] < (box_x + box_width) and current_ball.body.body_type == pymunk.Body.KINEMATIC:
            current_ball.body.position = pymunk.Vec2d(mouse[0], current_ball.body.position.y)

    # Ball dropping logic
    if frame_count == current_frames - 1:
        current_ball = spawn_new_ball(random.randint(0,4))
        ball_dropping = False
        current_frames = 90

    # End game conditions
    for ball in balls:
        if (ball.body.position.y - ball.radius < (box_y + 50)) and not ball_dropping:
            pygame.draw.line(screen, "white", (box_x, box_y + 10), (box_x + box_width, box_y + 20), 2)
        if (ball.body.position.y - ball.radius < (box_y + 10)) and not ball_dropping:
            show_game_over_screen()
            score = 0
            clear_balls(space, balls, current_ball)
            current_ball = spawn_new_ball(0)
            frame_count = 1
            break
    
    pygame.display.flip()
    dt = clock.tick(50) / 1000.0  # Update dt here (important for movement calculations)
    space.step(dt)  # Step the simulation
    frame_count = (frame_count + 1) % 50

pygame.quit()