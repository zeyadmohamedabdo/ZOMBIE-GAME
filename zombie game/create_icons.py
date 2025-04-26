import pygame
import os
import urllib.request
import io
from PIL import Image

# Initialize Pygame
pygame.init()

def download_and_save_image(url, save_path, size=(128, 128)):
    try:
        # Download the image
        with urllib.request.urlopen(url) as url_response:
            image_data = url_response.read()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert RGBA if needed
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Resize
        image = image.resize(size, Image.Resampling.LANCZOS)
        
        # Save
        image.save(save_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error downloading {save_path}: {e}")
        return False

def setup_assets():
    # Create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Dictionary of image URLs for each character
    image_urls = {
        'warrior': 'https://raw.githubusercontent.com/pixel-rpg/pixel-art/main/knight.png',  # Example URL
        'zombie1': 'https://raw.githubusercontent.com/pixel-rpg/pixel-art/main/stone-zombie.png',  # Example URL
        'zombie10': 'https://raw.githubusercontent.com/pixel-rpg/pixel-art/main/blood-zombie.png',  # Example URL
        'zombie100': 'https://raw.githubusercontent.com/pixel-rpg/pixel-art/main/demon-zombie.png',  # Example URL
        'castle_door': 'https://raw.githubusercontent.com/pixel-rpg/pixel-art/main/castle-door.png'  # Example URL
    }
    
    print("Please provide your own image files for:")
    print("1. warrior.png - A knight/warrior character")
    print("2. zombie1.png - A stone zombie")
    print("3. zombie10.png - A blood zombie")
    print("4. zombie100.png - A demon zombie")
    print("5. castle_door.png - A castle door")
    print("\nPlace these files in the 'assets' directory.")
    print("The images should be PNG files with transparent backgrounds.")
    print("Recommended size: 128x128 pixels")

# Create the assets directory and print instructions
setup_assets()

# Set up the surface for each icon (making them larger)
icon_size = 128  # Increased size
surface = pygame.Surface((icon_size, icon_size), pygame.SRCALPHA)

# Colors
GOLD = (255, 215, 0)
STONE = (120, 120, 120)  # Lighter stone color
BLOOD = (180, 0, 0)      # Brighter red
DEMON = (220, 0, 0)      # Brighter demon red
DOOR = (139, 69, 19)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SILVER = (192, 192, 192)

def create_warrior():
    surface.fill((0, 0, 0, 0))
    # Shield
    pygame.draw.circle(surface, SILVER, (45, 70), 20)
    pygame.draw.circle(surface, GOLD, (45, 70), 15)
    # Body (armor)
    pygame.draw.rect(surface, SILVER, (40, 60, 48, 48))
    # Head with helmet
    pygame.draw.circle(surface, SILVER, (64, 40), 20)
    pygame.draw.rect(surface, SILVER, (44, 30, 40, 20))
    # Face opening in helmet
    pygame.draw.ellipse(surface, BLACK, (54, 35, 20, 10))
    # Sword
    pygame.draw.rect(surface, SILVER, (75, 20, 8, 60))  # Blade
    pygame.draw.rect(surface, GOLD, (65, 70, 28, 8))    # Handle
    # Add outline
    pygame.draw.rect(surface, BLACK, (40, 60, 48, 48), 2)  # Body outline
    pygame.draw.circle(surface, BLACK, (64, 40), 20, 2)    # Helmet outline
    pygame.image.save(surface, 'assets/warrior.png')

def create_stone_zombie():
    surface.fill((0, 0, 0, 0))
    # Body
    pygame.draw.rect(surface, STONE, (40, 60, 48, 48))
    # Head
    pygame.draw.circle(surface, STONE, (64, 40), 20)
    # Eyes (more menacing)
    pygame.draw.ellipse(surface, BLACK, (50, 35, 10, 15))
    pygame.draw.ellipse(surface, BLACK, (68, 35, 10, 15))
    pygame.draw.ellipse(surface, (255, 0, 0), (52, 37, 6, 11))  # Red glow
    pygame.draw.ellipse(surface, (255, 0, 0), (70, 37, 6, 11))
    # Cracks in stone
    pygame.draw.line(surface, BLACK, (45, 70), (60, 85), 3)
    pygame.draw.line(surface, BLACK, (70, 65), (85, 80), 3)
    pygame.image.save(surface, 'assets/zombie1.png')

def create_blood_zombie():
    surface.fill((0, 0, 0, 0))
    # Body
    pygame.draw.rect(surface, BLOOD, (40, 60, 48, 48))
    # Blood drips
    for x in range(45, 85, 10):
        pygame.draw.polygon(surface, BLOOD, [(x, 108), (x+5, 118), (x-5, 118)])
    # Head
    pygame.draw.circle(surface, BLOOD, (64, 40), 20)
    # Eyes
    pygame.draw.ellipse(surface, BLACK, (50, 35, 12, 15))
    pygame.draw.ellipse(surface, BLACK, (70, 35, 12, 15))
    # Blood trails
    pygame.draw.line(surface, (120, 0, 0), (55, 35), (55, 50), 4)
    pygame.draw.line(surface, (120, 0, 0), (75, 35), (75, 50), 4)
    pygame.image.save(surface, 'assets/zombie10.png')

def create_demon_zombie():
    surface.fill((0, 0, 0, 0))
    # Body
    pygame.draw.rect(surface, DEMON, (40, 60, 48, 48))
    # Head
    pygame.draw.circle(surface, DEMON, (64, 40), 20)
    # Horns (larger and more detailed)
    pygame.draw.polygon(surface, BLACK, [(50, 30), (35, 10), (55, 25)])
    pygame.draw.polygon(surface, BLACK, [(78, 30), (93, 10), (73, 25)])
    # Eyes (glowing)
    pygame.draw.ellipse(surface, WHITE, (50, 35, 12, 15))
    pygame.draw.ellipse(surface, WHITE, (70, 35, 12, 15))
    pygame.draw.ellipse(surface, (255, 255, 0), (53, 38, 6, 9))  # Yellow glow
    pygame.draw.ellipse(surface, (255, 255, 0), (73, 38, 6, 9))
    # Demonic markings
    pygame.draw.line(surface, BLACK, (55, 70), (75, 90), 3)
    pygame.draw.line(surface, BLACK, (75, 70), (55, 90), 3)
    pygame.image.save(surface, 'assets/zombie100.png')

def create_castle_door():
    surface.fill((0, 0, 0, 0))
    # Stone frame
    pygame.draw.rect(surface, STONE, (30, 20, 68, 88))
    # Wooden door
    pygame.draw.rect(surface, DOOR, (35, 25, 58, 78))
    # Door patterns
    pygame.draw.rect(surface, (100, 50, 0), (40, 30, 48, 68))
    pygame.draw.line(surface, BLACK, (64, 30), (64, 98), 2)
    pygame.draw.line(surface, BLACK, (40, 64), (88, 64), 2)
    # Handle
    pygame.draw.circle(surface, GOLD, (75, 64), 6)
    pygame.draw.circle(surface, BLACK, (75, 64), 6, 1)
    # Stone texture
    for i in range(4):
        pygame.draw.line(surface, BLACK, (30, 20+i*29), (98, 20+i*29), 1)
    pygame.image.save(surface, 'assets/castle_door.png')

# Create all icons
create_warrior()
create_stone_zombie()
create_blood_zombie()
create_demon_zombie()
create_castle_door()

print("Enhanced icons created successfully in the assets directory!") 