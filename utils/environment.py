def is_on_colab():
    try:
        import google.colab
        return True
    except:
        return False
