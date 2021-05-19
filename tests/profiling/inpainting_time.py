import cProfile

from mmxai.text_removal.smart_text_removal import SmartTextRemover

def inpaint():
    remover = SmartTextRemover("../../mmxai/text_removal/frozen_east_text_detection.pb")
    img = remover.inpaint("https://www.iqmetrix.com/hubfs/Meme%2021.jpg")

def profile_inpainting():
    stats_name = "stats/" + "inpainting" + ".stats"
    cProfile.run(f"inpaint()", stats_name)

if __name__ == "__main__":
    profile_inpainting()
