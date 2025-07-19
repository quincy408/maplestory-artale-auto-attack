from chrome import ObjectDetection, MainScript

if __name__ == '__main__':
    user_name = 'user'
    monster_name = "mo_dragon"
    display_detection_result = False

    ObjectDetection.load_model(user_model_name=user_name, monster_model_name=monster_name)
    script = MainScript()
    script.main(display_detection_result=display_detection_result)