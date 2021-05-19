import cProfile

from mmxai.text_removal.smart_text_removal import SmartTextRemover

stats_folder = "./"

def inpaint():
    image_path = "https://www.iqmetrix.com/hubfs/Meme%2021.jpg"

    # Load the inpainter
    inpainter = SmartTextRemover("../../mmxai/text_removal/frozen_east_text_detection.pb")
    
    # Inpaint image
    img_inpainted = inpainter.inpaint(image_path)

def profile_inpaint():
    stats_name = "smart_inpainter.stats"
    stats_name = stats_folder + stats_name
    cProfile.run("inpaint()", stats_name)

if __name__ == "__main__":
    profile_inpaint()