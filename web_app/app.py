from flask import Flask, render_template, request, url_for, redirect, session, flash
from interpretability4mmf import shap_mmf, lime_mmf, torchray_mmf
from mmxai.text_removal.smart_text_removal import SmartTextRemover
import app_utils as ut
import file_manage as fm
from config import APSchedulerJobConfig
from datetime import datetime, timedelta
from flask_apscheduler import APScheduler
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__)
app.config.from_object(APSchedulerJobConfig)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "Secret Key"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "Secret Key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///user_info.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
app.config["SQLALCHEMY_ECHO"] = True
db = SQLAlchemy(app)

# constant dictionary
EXAMPLES = ut.read_examples_metadata()


class user_info(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(20))
    ip_addr = db.Column(db.String(20))
    expired_time = db.Column(db.DateTime)

    def __init__(self, file_name, ip_addr, expired_time):
        self.file_name = file_name
        self.ip_addr = ip_addr
        self.expired_time = expired_time


@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(hours = 1)
    if session.get('user') is None:
        ip_addr = request.remote_addr
        #ip_addr=request.headers['X-Real-Ip']
        total_user = user_info.query.filter_by(ip_addr=ip_addr).all()
        if len(total_user) > 3:
            user_delete = user_info.query.filter_by(ip_addr=ip_addr).first()
            db.session.delete(user_delete)
            db.session.commit()
        user_id = fm.generate_random_str(8)
        fm.mkdir('./static/user/' + user_id)
        session['user'] = user_id
        file_name = session['user']
        expired_time = datetime.now() + timedelta(days=1)
        user_insert = user_info(file_name, ip_addr, expired_time)
        db.session.add(user_insert)
        db.session.commit()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/docs/")
def docs():
    return render_template("docsify.html")


@app.route("/explainers/hateful-memes")
def hateful_memes():
    img_name = session.get("imgName")
    img_text = session.get("imgText")
    img_exp = session.get("imgExp")
    text_exp = session.get("textExp")
    cls_result = session.get("clsResult")
    txt_msg = session.get("txtMsg")
    img_msg = session.get("imgMsg")
    user_model = session.get("userModel")
    model_type = session.get("modelType")
    model_path = session.get("modelPath")

    img_name_eg = session.get("imgNameEg")
    img_text_eg = session.get("imgTextEg")
    img_exp_eg = session.get("imgExpEg")
    text_exp_eg = session.get("textExpEg")
    cls_result_eg = session.get("clsResultEg")
    txt_msg_eg = session.get("txtMsgEg")
    img_msg_eg = session.get("imgMsgEg")
    model_type_eg = session.get("modelTypeEg")

    if model_path is not None:
        model_name = model_path.split("/")[-1]
        # only display part of the filename if the filename is longer than 6 chars
        name, extension = os.path.splitext(model_name)
        if len(name) > 6:
            model_name = name[0:3] + "..." + name[-1] + extension
    else:
        model_name = None

    if user_model == "no_model":
        cur_opt = "MMF ({}) with pretrained checkpoint".format(model_type)
    elif user_model == "mmf":
        cur_opt = "MMF ({}) with user checkpoint".format(model_type)
    elif user_model == "onnx":
        cur_opt = "User uploaded ONNX model"
    else:
        cur_opt = None

    if model_type_eg is not None:
        cur_model_eg = "MMF ({}) with pretrained checkpoint".format(model_type_eg)
    else:
        cur_model_eg = None

    return render_template(
        "explainers/hateful-memes.html",
        imgName=img_name,
        imgText=img_text,
        imgExp=img_exp,
        textExp=text_exp,
        clsResult=cls_result,
        txtMsg=txt_msg,
        imgMsg=img_msg,
        curOption=cur_opt,
        fileName=model_name,

        imgNameEg=img_name_eg,
        imgTextEg=img_text_eg,
        imgExpEg=img_exp_eg,
        textExpEg=text_exp_eg,
        clsResultEg=cls_result_eg,
        txtMsgEg=txt_msg_eg,
        imgMsgEg=img_msg_eg,
        curModelEg=cur_model_eg,
    )


@app.route("/uploadImage", methods=["POST"])
def upload_image():
    try:
        file = request.files["inputImg"]
        img_name = "user/" + session["user"] + "/" + file.filename
        img_path = "static/" + img_name
        file.save(img_path)
        ut.check_image(img_path)
    except ut.InputError as e:
        msg = e.message()
        flash(msg, "danger")
        return redirect(url_for("hateful_memes"))
    except:
        flash("Please select a file before continuing", "danger")
        return redirect(url_for("hateful_memes"))
    session["imgName"] = img_name
    session["imgText"] = None
    session["imgExp"] = None
    session["textExp"] = None
    session["clsResult"] = None
    return redirect(url_for("hateful_memes"))


@app.route("/uploadModel", methods=["POST"])
def upload_model():
    try:
        option = request.form["selectOption"]
    except:
        flash("Please select an option before continuing", "danger")
        return redirect(url_for("hateful_memes"))

    if option == "internalModel":
        try:
            selected_model = request.form["selectModel"]
            file = request.files["inputCheckpoint1"]
        except:
            flash("Please select a file before continuing", "danger")
            return redirect(url_for("hateful_memes"))

        if not file.filename.split(".")[-1] == "ckpt":
            flash(
                "Sorry, MMXAI only supports .ckpt checkpoints saved directly from mmf trainer, please check your file type",
                "danger",
            )
            return redirect(url_for("hateful_memes"))

        file_path = "user/" + session["user"] + "/" + file.filename
        session["modelPath"] = file_path
        session["modelType"] = selected_model
        session["userModel"] = "mmf"
        file.save("static/" + file_path)

    elif option == "selfModel":
        try:
            tokenizer = request.form["tokenizer"]
            file = request.files["inputCheckpoint2"]
        except:
            flash("Please select a file before continuing", "danger")
            return redirect(url_for("hateful_memes"))

        if not file.filename.split(".")[-1] == "onnx":
            flash(
                "Sorry, MMXAI only supports .onnx checkpoints, please check your file type",
                "danger",
            )
            return redirect(url_for("hateful_memes"))

        file_path = "user/" + session["user"] + "/" + file.filename
        session["modelPath"] = file_path
        session["modelType"] = tokenizer
        session["userModel"] = "onnx"
        file.save("static/" + file_path)

    elif option == "noModel":
        selected_existing_model = request.form["selectExistingModel"]
        session["modelType"] = selected_existing_model
        session["modelPath"] = None
        session["userModel"] = "no_model"
    else:
        raise Exception("Sorry, you must select an option to continue!!!")
    return redirect(url_for("hateful_memes"))


@app.route("/inpaint")
def inpaint():
    img_name = session.get("imgName")
    # Prepare the image path and names
    folder_path = "static/"

    # Must upload image before performing inpainting
    try:
        image_path = folder_path + img_name
    except:
        flash(
            "Please upload your image before trying to remove the texts from it",
            "danger",
        )
        return redirect(url_for("hateful_memes"))

    img_name_no_extension = os.path.splitext(img_name)[0]
    img_extension = os.path.splitext(img_name)[1]

    inpainted_image_name = img_name_no_extension + "_inpainted" + img_extension
    save_path = folder_path + inpainted_image_name
    if not os.path.isfile(save_path):
        # Load the inpainter
        try:
            inpainter = SmartTextRemover(
                "../mmxai/text_removal/frozen_east_text_detection.pb"
            )
        except:
            flash("Sorry, cannot load inpainter", "danger")
            return redirect(url_for("hateful_memes"))
        # Inpaint image
        try:
            img_inpainted = inpainter.inpaint(image_path)
        except:
            flash("Sorry, cannot inpaint image", "danger")
            return redirect(url_for("hateful_memes"))
        # save inpainted image
        img_inpainted.save(save_path)

    session["imgName"] = inpainted_image_name
    return redirect(url_for("hateful_memes"))


@app.route("/restoreImage")
def restore():
    try:
        img_path = session["imgName"]
    except:
        flash("Please upload and click inpaint your image before restoring", "danger")
        return redirect(url_for("hateful_memes"))

    root, extension = os.path.splitext(img_path)
    if "_inpainted" in root:
        img_path = root[:-10] + extension
        session["imgName"] = img_path
    return redirect(url_for("hateful_memes"))


@app.route("/explainers/hateful-memes/predict", methods=["POST"])
def predict():
    # ensure having images and texts to predict

    try:    # check text
        img_text = ut.check_text(request.form["texts"])
    except ut.InputError as e:
        msg = e.message()
        flash(msg, "danger")
        return redirect(url_for("hateful_memes"))

    try:
        exp_method = request.form["expMethod"]
        exp_inclination = request.form["expInc"]

        shap_algo = request.form["shapAlgo"]
        shap_max_eval = request.form["shapMaxEval"]
        shap_batch_size = request.form["shapBatchSize"]

        lime_sample_number = request.form["limeSamplesNumber"]

        torchray_max_iteration = request.form["torchrayMaxIteration"]

        user_model = session.get("userModel")
        img_name = session.get("imgName")
        assert img_name is not None

        model_type = session.get("modelType")
        model_path = session.get("modelPath")
    except:
        flash("Please finish the uploading process before predicting", "danger")
        return redirect(url_for("hateful_memes"))

    # ensure load model correctly
    try:
        model, label_to_explain, cls_label, cls_confidence = ut.prepare_explanation(
            img_name, img_text, user_model, model_type, model_path, exp_inclination
        )
    except ut.InputError as e:
        msg = e.message()
        flash(msg, "danger")
        return redirect(url_for("hateful_memes"))

    hateful = "HATEFUL" if cls_label == 1 else "NON-HATEFUL"
    cls_result = (
        f"Your uploaded image and text combination "
        f"looks like a <strong>{hateful}</strong> meme, with {cls_confidence * 100: .2f}% confidence. "
    )

    try:
        if exp_method == "shap":
            text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
                img_name,
                img_text,
                model,
                label_to_explain,
                ut.model_output(cls_label, cls_confidence),
                algorithm=shap_algo,
                max_evals=int(shap_max_eval),
                batch_size=int(shap_batch_size),
                segment_algo="slic",
                n_img_segments=100,
                sigma=0,
            )
        elif exp_method == "lime":
            text_exp, img_exp, txt_msg, img_msg = lime_mmf.lime_multimodal_explain(
                img_name,
                img_text,
                model,
                label_to_explain,
                num_samples=int(lime_sample_number),
            )
        elif exp_method == "torchray":
            (
                text_exp,
                img_exp,
                txt_msg,
                img_msg,
            ) = torchray_mmf.torchray_multimodal_explain(
                img_name,
                img_text,
                model,
                label_to_explain,
                max_iteration=int(torchray_max_iteration),
            )
        else:
            text_exp, img_exp, txt_msg, img_msg = shap_mmf.shap_multimodal_explain(
                img_name,
                img_text,
                model,
                label_to_explain,
                ut.model_output(cls_label, cls_confidence),
                algorithm=shap_algo,
                max_evals=int(shap_max_eval),
                batch_size=int(shap_batch_size),
                segment_algo="slic",
                n_img_segments=100,
                sigma=0,
            )
    except Exception as e:
        flash(
            f"Some uncaught error occurred while explaining using {exp_method}: {e}",
            "danger",
        )
        return redirect(url_for("hateful_memes"))

    session["clsResult"] = cls_result
    session["imgText"] = img_text
    session["textExp"] = text_exp
    session["imgExp"] = img_exp

    exp_text_visl, _ = os.path.splitext(img_exp)
    exp_text_visl = exp_text_visl[:-4] + "_txt.png"

    session["txtMsg"] = txt_msg
    session["imgMsg"] = img_msg

    try:
        ut.text_visualisation(text_exp, cls_label, exp_text_visl)
        session["textExp"] = exp_text_visl
    except:
        flash(
            "Unable to plot text explanation visualisation because output text explanation is empty",
            "danger",
        )
        session["textExp"] = None

    flash(
        "Done! Hover over the output images to see how to interpret the results",
        "success",
    )
    return redirect(url_for("hateful_memes"))


@app.route("/selectExample", methods=["POST"])
def select_example():
    example_id = request.form["exampleID"]
    example = EXAMPLES.get(example_id)
    img_name = example.get("imgName")
    img_text = example.get("imgTexts")

    session["exampleID"] = example_id
    session["imgNameEg"] = img_name
    session["imgTextEg"] = img_text
    session["imgExpEg"] = None
    session["textExpEg"] = None
    session["clsResultEg"] = None
    return redirect(url_for("hateful_memes"))


@app.route("/fetchExample", methods=["POST"])
def fetch_example():
    try:
        method = request.form["expMethod"]
        assert method is not None
        example_id = session.get("exampleID")
        assert example_id is not None
    except AssertionError as e:
        flash("Please select an example and method before explaining", "danger")
        return redirect(url_for("hateful_memes"))

    example = EXAMPLES.get(example_id)
    info = example.get(method)

    session["clsResultEg"] = example.get("clsResult")
    session["modelTypeEg"] = info.get("modelType")
    session["textExpEg"] = info.get("txtExp")
    session["imgExpEg"] = info.get("imgExp")
    session["txtMsgEg"] = info.get("txtMsg")
    session["imgMsgEg"] = info.get("imgMsg")

    return redirect(url_for("hateful_memes"))


if __name__ == "__main__":
    app.run(debug=True)

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()
