function exampleSwitch() {
  if (document.getElementById("exampleSwitch").checked === true) {
    document.getElementById("ctrlBar").style.display = "none";
    document.getElementById("ctrlBarEg").style.display = "block";
    document.getElementById("infArea").style.display = "none";
    document.getElementById("infAreaEg").style.display = "block";
    document.getElementById("imgInput").style.display = "none";
    document.getElementById("imgInputEg").style.display = "block";
    document.getElementById("imgText").style.display = "none";
    document.getElementById("imgTextEg").style.display = "block";
    document.getElementById("imgExp").style.display = "none";
    document.getElementById("imgExpEg").style.display = "block";
    if (document.getElementById("txtExp"))
      document.getElementById("txtExp").style.display = "none";
    if (document.getElementById("txtExpEg"))
      document.getElementById("txtExpEg").style.display = "block";
  } else {
    document.getElementById("ctrlBarEg").style.display = "none";
    document.getElementById("ctrlBar").style.display = "block";
    document.getElementById("infAreaEg").style.display = "none";
    document.getElementById("infArea").style.display = "block";
    document.getElementById("imgInputEg").style.display = "none";
    document.getElementById("imgInput").style.display = "block";
    document.getElementById("imgTextEg").style.display = "none";
    document.getElementById("imgText").style.display = "block";
    document.getElementById("imgExpEg").style.display = "none";
    document.getElementById("imgExp").style.display = "block";
    if (document.getElementById("txtExpEg")) {
      document.getElementById("txtExpEg").style.display = "none";
    }
    if (document.getElementById("txtExp"))
      document.getElementById("txtExp").style.display = "block";
  }
  console.log("mode stored as", document.getElementById("exampleSwitch").checked)
  localStorage.setItem(
      "mode",
      document.getElementById("exampleSwitch").checked
  );
}

var myDefaultAllowList = bootstrap.Tooltip.Default.allowList
myDefaultAllowList.span = ['style']
myDefaultAllowList.p = ['style']

var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
  return new bootstrap.Popover(popoverTriggerEl)
})

var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})

var shapMaxEvalSlider = document.getElementById("shapEvalSlider");
var shapMaxEvalValue = document.getElementById("shapEvalValue");
shapMaxEvalSlider.oninput = function () {
  shapMaxEvalValue.innerHTML = this.value;
}

var shapBatchSizeSlider = document.getElementById("shapBatchSlider");
var shapBatchSizeValue = document.getElementById("shapBatchValue");
shapBatchSizeSlider.oninput = function () {
  shapBatchSizeValue.innerHTML = this.value;
}

var limeSamplesNumberSlider = document.getElementById("limeSamplesNumberSlider");
var limeSamplesNumberValue = document.getElementById("limeSamplesNumberValue");
limeSamplesNumberSlider.oninput = function () {
  limeSamplesNumberValue.innerHTML = this.value;
}

var torchrayMaxIterationSlider = document.getElementById("torchrayMaxIterationSlider");
var torchrayMaxIterationValue = document.getElementById("torchrayMaxIterationValue");
torchrayMaxIterationSlider.oninput = function () {
  torchrayMaxIterationValue.innerHTML = this.value;
}

function configParam() {
  var shapParamModal = new bootstrap.Modal(document.getElementById("customiseShap"))
  var limeParamModal = new bootstrap.Modal(document.getElementById("customiseLime"))
  var torchrayParamModal = new bootstrap.Modal(document.getElementById("customiseTorchray"))
  console.log("shap() triggered")
  if (document.getElementById("selectMethod").value === "shap") {
    shapParamModal.show()
  } else if (document.getElementById("selectMethod").value === "lime") {
    limeParamModal.show()
  } else if (document.getElementById("selectMethod").value === "torchray") {
    torchrayParamModal.show()
  }
}

function explainingEg() {
  localStorage.setItem(
      "methodEg",
      document.getElementById("selectMethodEg").value
  );
  localStorage.setItem(
      "inclinationEg",
      document.getElementById("selectIncEg").value
  );
}

function explaining() {
  localStorage.setItem(
      "method",
      document.getElementById("selectMethod").value
  );
  localStorage.setItem(
      "inclination",
      document.getElementById("selectInc").value
  );
  localStorage.setItem(
      "shapAlgo",
      document.getElementById("selectShapAlgo").value
  );
  localStorage.setItem(
      "shapMaxEval",
      document.getElementById("shapEvalSlider").value
  );
  localStorage.setItem(
      "shapBatchSize",
      document.getElementById("shapBatchSlider").value
  );
  localStorage.setItem(
      "limeSamplesNumber",
      document.getElementById("limeSamplesNumberSlider").value
  );
  localStorage.setItem(
      "torchrayMaxIteration",
      document.getElementById("torchrayMaxIterationSlider").value
  );

  document.getElementById("loader").style.display = "block";
  if (document.getElementById("imgExp")) {
    document.getElementById("imgExp").style.display = "none";
  }
  if (document.getElementById("imgExp")) {
    document.getElementById("imgExp").style.display = "none";
  }
  if (document.getElementById("txtExp")) {
    document.getElementById("txtExp").style.display = "none";
  }
  document.getElementById("infArea").innerHTML = "Explaining...please wait...";
  document.getElementById("exampleSwitch").setAttribute("disabled", "true")
}

function stashTextInput() {
  console.log("txt changed")
  localStorage.setItem(
      "txtInput",
      document.getElementById("imgText").value
  );
}

function resetLocalStorage() {
  localStorage.removeItem("method");
  localStorage.removeItem("txtInput");
  localStorage.removeItem("inclination");
}

function showOptionForm(that) {
  if (that.value === "internalModel") {
    document.getElementById("internalModel").style.display = "block";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "none";
  } else if (that.value === "selfModel") {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "block";
    document.getElementById("noModel").style.display = "none";
  } else if (that.value === "noModel") {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "block";
  } else {
    document.getElementById("internalModel").style.display = "none";
    document.getElementById("selfModel").style.display = "none";
    document.getElementById("noModel").style.display = "none";
  }
}

window.addEventListener("DOMContentLoaded", (event) => {
  const mode = localStorage.getItem("mode");
  if (mode) {
    if (mode === "true") {
      console.log("checked")
      document.getElementById("exampleSwitch").setAttribute("checked", "true");
    } else {
      console.log("unchecked")
    }
    console.log("mode change to", mode)
  }
  exampleSwitch();

  const selectedMethodEg = localStorage.getItem("methodEg");
  if (selectedMethodEg) {
    document.getElementById("selectMethodEg").value = selectedMethodEg;
  }
  const selectIncEg = localStorage.getItem("inclinationEg");
  if (selectIncEg) {
    document.getElementById("selectIncEg").value = selectIncEg;
  }
  const txtInput = localStorage.getItem("txtInput");
  if (txtInput) {
    document.getElementById("imgText").value = txtInput;
  }
  const selectedMethod = localStorage.getItem("method");
  if (selectedMethod) {
    document.getElementById("selectMethod").value = selectedMethod;
  }
  const selectInc = localStorage.getItem("inclination");
  if (selectInc) {
    document.getElementById("selectInc").value = selectInc;
  }
  const selectedShapAlgo = localStorage.getItem("shapAlgo");
  if (selectedShapAlgo) {
    document.getElementById("selectShapAlgo").value = selectedShapAlgo;
  } else {
    document.getElementById("selectShapAlgo").value = 'partition';
  }
  const selectedShapMaxEval = localStorage.getItem("shapMaxEval");
  if (selectedShapMaxEval) {
    document.getElementById("shapEvalSlider").value = selectedShapMaxEval;
  } else {
    document.getElementById("shapEvalSlider").value = '300';
  }
  const selectedShapBatchSize = localStorage.getItem("shapBatchSize");
  if (selectedShapBatchSize) {
    document.getElementById("shapBatchSlider").value = selectedShapBatchSize;
  } else {
    document.getElementById("shapBatchSlider").value = '50';
  }
  const selectedLimeSamplesNumber = localStorage.getItem("limeSamplesNumber");
  if (selectedLimeSamplesNumber) {
    document.getElementById("limeSamplesNumberSlider").value = selectedLimeSamplesNumber;
  } else {
    document.getElementById("limeSamplesNumberSlider").value = '500';
  }
  const selectedTorchrayMaxIteration = localStorage.getItem("torchrayMaxIteration");
  if (selectedTorchrayMaxIteration) {
    document.getElementById("torchrayMaxIterationSlider").value = selectedTorchrayMaxIteration;
  } else {
    document.getElementById("torchrayMaxIterationSlider").value = '800';
  }

  shapMaxEvalValue.innerHTML = shapMaxEvalSlider.value;
  shapBatchSizeValue.innerHTML = shapBatchSizeSlider.value;
  limeSamplesNumberValue.innerHTML = limeSamplesNumberSlider.value;
  torchrayMaxIterationValue.innerHTML = torchrayMaxIterationSlider.value;
});

let tut = document.querySelector("#btnTutorial");
tut.onclick = function () {
  setTimeout(() => {
    const driver = new Driver();
    const stepList = [
      {
        element: "#btnUploadMdl",
        popover: {
          title: "Step 1",
          description:
              "Click here to select a model or upload your own checkpoints",
        },
      },
      {
        element: "#btnUploadImg",
        popover: {
          title: "Step 2",
          description: "Click here to select an image and upload",
        },
      },
      {
        element: "#imgBox1",
        popover: {
          title: "Then",
          description:
              "After uploading, your selected image will be displayed here",
        },
      },
      {
        element: "#imgText",
        popover: {
          title: "Step 3",
          description:
              "After uploading the image, add some texts to the image here",
        },
      },
      {
        element: "#btnInpaint",
        popover: {
          title: "Next: An Optional Choice, Text Removal",
          description:
              "If you want to remove texts that appears on the uploaded image, click this button",
        },
      },
      {
        element: "#selectMethod",
        popover: {
          title: "Step 4",
          description: "Select a method for explaining the model prediction",
        },
      },
      {
        element: "#configureMethod",
        popover: {
          title: "Step 5",
          description: "You may also configure the parameters to those methods",
        },
      },
      {
        element: "#selectInc",
        popover: {
          title: "Step 6",
          description: "If you want to see what factors in image/texts support model's prediction, select 'Encourage'. If you want to see what factors are against model's prediction, select 'Discourage'",
        },
      },
      {
        element: "#btnPredict",
        popover: {
          title: "Step 7",
          description: "Press this button to run the explanation algorithm",
        },
      },
      {
        element: "#imgBox2",
        popover: {
          title: "Finally",
          description: "It may take a few minutes to run the explanation algorithm. The result image will be shown here.",
        },
      },
    ];
    driver.defineSteps(stepList);
    driver.start();
  }, 50);
};

let tutEg = document.querySelector("#btnTutorialEg");
tutEg.onclick = function () {
  setTimeout(() => {
    const driver = new Driver();
    const stepList = [
      {
        element: "#btnSelExampleImg",
        popover: {
          title: "Step 1",
          description:
              "Click here to select a meme from the example library",
        },
      },
      {
        element: "#selectMethodEg",
        popover: {
          title: "Step 2",
          description: "Select a explanation method that you want to see explanations from",
        },
      },
      {
        element: "#btnExplainEg",
        popover: {
          title: "Step 3",
          description: "Press this button to retrieve the explanation results generated by the chosen method in step 2",
        },
      },
      {
        element: "#imgBox2",
        popover: {
          title: "Finally",
          description: "After clicking Explain button, the explanation results will instantly be shown here",
        },
      },
    ];
    driver.defineSteps(stepList);
    driver.start();
  }, 50);
};