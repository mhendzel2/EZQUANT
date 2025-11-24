#!/usr/bin/env python3
"""
Create application icon for Nuclei Segmentation App
Generates a simple icon with nucleus-like appearance

Usage:
    python create_icon.py
    
Output:
    icon.ico (256x256 multi-resolution icon)
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_nucleus_icon(size=256):
    """Create a nucleus-like icon"""
    # Create white background
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Define colors (modern blue palette)
    outer_fill = '#2E86AB'      # Dark blue
    outer_outline = '#0F4C75'   # Darker blue
    inner_fill = '#6C9BD1'      # Light blue
    inner_outline = '#2E86AB'   # Medium blue
    chromatin = '#A23B72'       # Purple for chromatin
    
    # Calculate dimensions
    margin = size // 8
    outer_radius = size - 2 * margin
    inner_radius = size // 3
    
    # Draw outer nucleus membrane
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=outer_fill,
        outline=outer_outline,
        width=max(2, size // 32)
    )
    
    # Draw inner nucleolus
    center = size // 2
    nucleolus_radius = inner_radius // 2
    draw.ellipse(
        [
            center - nucleolus_radius,
            center - nucleolus_radius,
            center + nucleolus_radius,
            center + nucleolus_radius
        ],
        fill=inner_fill,
        outline=inner_outline,
        width=max(1, size // 64)
    )
    
    # Add chromatin-like spots
    chromatin_spots = [
        (center - inner_radius, center - inner_radius // 2),
        (center + inner_radius // 2, center - inner_radius // 3),
        (center - inner_radius // 3, center + inner_radius // 2),
        (center + inner_radius // 3, center + inner_radius // 3),
    ]
    
    spot_size = max(2, size // 20)
    for x, y in chromatin_spots:
        draw.ellipse(
            [x - spot_size, y - spot_size, x + spot_size, y + spot_size],
            fill=chromatin
        )
    
    return img

def create_multisize_icon(output_path='icon.ico'):
    """Create multi-resolution icon file"""
    sizes = [256, 128, 64, 48, 32, 16]
    images = []
    
    print("Creating icon with multiple resolutions...")
    for size in sizes:
        img = create_nucleus_icon(size)
        images.append(img)
        print(f"  ✓ Created {size}x{size} version")
    
    # Save as ICO with all resolutions
    images[0].save(
        output_path,
        format='ICO',
        sizes=[(s, s) for s in sizes],
        append_images=images[1:]
    )
    
    print(f"\n✅ Icon saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Also save as PNG for reference
    png_path = output_path.replace('.ico', '.png')
    images[0].save(png_path, format='PNG')
    print(f"✅ PNG version saved to: {png_path}")

def create_simple_text_icon(output_path='icon.ico', text='N'):
    """Create simple text-based icon as fallback"""
    size = 256
    img = Image.new('RGB', (size, size), color='#2E86AB')
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", size // 2)
    except:
        font = ImageFont.load_default()
    
    # Draw letter 'N' for Nuclei
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    
    draw.text((x, y), text, fill='white', font=font)
    
    # Save
    img.save(output_path, format='ICO', sizes=[(256, 256)])
    print(f"✅ Simple text icon saved to: {output_path}")

if __name__ == '__main__':
    print("=" * 60)
    print("Nuclei Segmentation App - Icon Generator")
    print("=" * 60)
    
    try:
        # Try to create detailed nucleus icon
        create_multisize_icon('icon.ico')
        print("\n✨ Icon created successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Error creating detailed icon: {e}")
        print("Creating simple fallback icon...")
        try:
            create_simple_text_icon('icon.ico')
        except Exception as e2:
            print(f"❌ Failed to create icon: {e2}")
            print("You can use any 256x256 ICO file as icon.ico")
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Verify icon.ico was created")
    print("  2. Run: pyinstaller build.spec")
    print("=" * 60)
