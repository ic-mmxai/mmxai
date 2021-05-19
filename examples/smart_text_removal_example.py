from mmxai.text_removal.smart_text_removal import SmartTextRemover


def main(model_path=None, show_img=True):
    if not model_path:
        model_path = "mmxai/text_removal/frozen_east_text_detection.pb"
    
    remover = SmartTextRemover(model_path)
    
    img = remover.inpaint(
        "https://www.iqmetrix.com/hubfs/Meme%2021.jpg")
    
    if show_img:
        img.show()

if __name__ == "__main__":
    main()