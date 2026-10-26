# 🔥 WildfireWatch

### Satellite-Based Wildfire and Industrial Heat Source Classification Using PCA and CNN

WildfireWatch is a machine learning project that explores the classification of satellite-detected thermal anomalies into potential wildfire events and industrial heat sources. By combining Principal Component Analysis (PCA) with a Convolutional Neural Network (CNN), the project investigates how thermal patterns can be analysed to support wildfire monitoring and environmental assessment.

The project uses NASA FIRMS (Fire Information for Resource Management System) hotspot data and geographical information about power plants to provide context for identifying and interpreting thermal anomalies.

## 🌍 Problem Statement

Satellite-based fire detection systems identify thermal anomalies across large geographical areas. However, not every detected hotspot represents a wildfire. Industrial facilities, power plants, and other heat-emitting sources can also produce thermal signatures that resemble fire activity.

Distinguishing between these sources is an important challenge in satellite-based fire monitoring. Misclassifying industrial heat as a wildfire may contribute to false alarms, while failing to identify a genuine wildfire may delay further investigation.

WildfireWatch explores the use of machine learning and satellite-derived thermal information to help differentiate potential wildfire activity from industrial heat sources.

## 🎯 Objectives

- Analyse satellite-detected thermal anomalies using machine learning.
- Apply PCA to learn compact representations of thermal patterns and analyse reconstruction errors.
- Use CNN-based image analysis to identify spatial features in thermal inputs.
- Combine PCA-based and CNN-based scores into an ensemble prediction.
- Use geographical proximity to known power plants as supporting context for thermal anomaly analysis.
- Explore the application of image-based machine learning to wildfire monitoring.
- Investigate the potential of automated classification to support environmental observation and decision-making.

## 🛰️ Data Sources

### 1. NASA FIRMS — Fire Information for Resource Management System

NASA FIRMS provides satellite-derived fire and thermal anomaly detections from instruments including the Visible Infrared Imaging Radiometer Suite (VIIRS).

The project uses FIRMS data to analyse detected hotspots and their associated attributes, including:

- Latitude and longitude
- Acquisition date and time
- Brightness temperature
- Fire Radiative Power (FRP)
- Detection confidence
- Day or night observation
- Satellite and instrument information

These attributes provide information about the location, timing, and thermal characteristics of detected anomalies.

FIRMS detections indicate potential fire or thermal activity; they do not independently confirm that an event is a wildfire.

**Source:** https://firms.modaps.eosdis.nasa.gov/

### 2. VIIRS Satellite Imagery

VIIRS imagery products can provide spatial thermal information for analysing hotspot patterns.

The relevant NASA data products include:

- **VNP02IMG:** Calibrated image radiances from the VIIRS instrument.
- **VNP03IMG:** Geolocation information associated with VIIRS image observations.

When matched imagery and geolocation files are available, they can support the extraction of image patches corresponding to thermal anomalies. These image patches can then be used as inputs for image-based analysis.

The distinction between actual satellite imagery and synthetic thermal representations is important: synthetic inputs can support experimentation, but they do not reproduce all the spatial and spectral information contained in real satellite observations.

**Source:** https://ladsweb.modaps.eosdis.nasa.gov/

### 3. Global Power Plant Database

Power plant location data provides geographical context for analysing hotspots near known industrial infrastructure.

The project uses the coordinates of power plants to calculate the proximity of detected thermal anomalies to these facilities. This information can help investigate whether a hotspot may be associated with an industrial source.

Proximity is treated as supporting geographical evidence rather than definitive proof. A nearby hotspot is not necessarily industrial, and a hotspot far from a power plant is not automatically a wildfire.

**Source:** https://datasets.wri.org/

## 🧠 Machine Learning Methodology

### 1. Principal Component Analysis (PCA)

Principal Component Analysis is a dimensionality-reduction technique that transforms high-dimensional data into a smaller set of components while retaining important patterns of variation.

In WildfireWatch, PCA is used to represent thermal inputs in a lower-dimensional feature space.

The PCA workflow involves:

1. Preparing and standardising the input representation where appropriate.
2. Learning principal components from the training data.
3. Projecting thermal inputs into the learned component space.
4. Reconstructing the inputs using the retained components.
5. Calculating reconstruction error to measure the difference between the original and reconstructed representations.

Reconstruction error provides a measure of how well a thermal pattern is represented by the PCA model. Its usefulness for distinguishing wildfire and industrial heat sources depends on the training data and the relationship between reconstruction behaviour and the target classes.

### 2. Convolutional Neural Network (CNN)

A Convolutional Neural Network is a deep learning architecture designed to learn spatial features from image data.

Unlike approaches that rely only on manually selected numerical attributes, a CNN can learn local patterns and spatial relationships directly from image inputs.

In WildfireWatch, the CNN is used to:

- Process thermal image representations.
- Learn spatial features associated with the training examples.
- Identify patterns relevant to the target classification.
- Generate a model score or probability for the predicted class.

The quality of CNN predictions depends on the representativeness of the training data, the validity of the input imagery, and independent model evaluation.

### 3. PCA + CNN Ensemble

WildfireWatch combines PCA-based analysis with CNN-based classification to form an ensemble prediction.

The two methods provide different types of information:

- **PCA** analyses the representation and reconstruction behaviour of thermal patterns.
- **CNN** learns spatial features from image inputs.

In the configured ensemble, the PCA-based score contributes 40% and the CNN probability contributes 60% to the final score.

The combination is expressed as:

`Final Score = (0.40 × PCA Score) + (0.60 × CNN Probability)`

The final score is used by the classification pipeline to determine the predicted class according to its configured decision threshold.

The interpretation of this score depends on how the PCA score is transformed or calibrated and how the target classes are encoded. A weighted score should not automatically be interpreted as a calibrated probability of a real-world wildfire.

## 🔄 Project Workflow

1. **Satellite data preparation:** Prepare FIRMS hotspot records and relevant geographical information.
2. **Thermal input preparation:** Obtain suitable image patches or the thermal representations supported by the current pipeline.
3. **PCA analysis:** Project inputs into a reduced-dimensional representation and calculate reconstruction-based scores.
4. **CNN inference:** Process the image inputs to obtain learned spatial features and classification scores.
5. **Ensemble prediction:** Combine the PCA and CNN scores using the configured weights.
6. **Classification:** Apply the configured decision rule to obtain the predicted category.
7. **Geographical interpretation:** Use power plant proximity as supporting context when investigating potential industrial heat sources.

The final classification pipeline is based on PCA and CNN. FIRMS attributes and power plant coordinates support data preparation and geographical analysis rather than acting as separate prediction models in the ensemble.

## 🔥 Why Distinguish Wildfires from Industrial Heat Sources?

Distinguishing these sources can improve the interpretation of satellite thermal observations.

**Potential practical benefits include:**

- **Fire monitoring:** Help analysts prioritise thermal anomalies that may require further investigation.
- **Reduced ambiguity:** Provide an additional classification signal when interpreting hotspots near industrial infrastructure.
- **Environmental assessment:** Support the investigation of possible wildfire activity across large geographical regions.
- **Resource prioritisation:** Help direct human attention towards detections that warrant closer examination.
- **Satellite data analysis:** Demonstrate how machine learning can complement conventional hotspot detection products.

These are intended applications, not guarantees that the model will reduce false alarms or detect every wildfire. Such benefits must be established through appropriate evaluation.

## 📊 Model Evaluation

A reliable evaluation should assess how well the system distinguishes the target classes on data that were not used to train the models.

Useful evaluation metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted wildfire cases that are actually wildfire cases.
- **Recall:** Proportion of actual wildfire cases correctly identified.
- **F1-score:** Balance between precision and recall.
- **Confusion matrix:** Breakdown of correct and incorrect predictions for each class.

Class-wise evaluation is particularly important because false wildfire alarms and missed wildfires have different practical consequences.

The evaluation should also use independently labelled examples and, where possible, geographically and temporally separate test data. Performance metrics should only be reported when supported by actual experimental results.

## ⚠️ Limitations

- Satellite thermal anomalies can originate from multiple natural and human-made sources.
- Power plant proximity alone cannot establish the cause of a thermal detection.
- Synthetic thermal representations may not capture the full characteristics of real satellite imagery.
- PCA reconstruction error does not inherently identify a wildfire; its classification value must be demonstrated empirically.
- CNN predictions depend on the quality, quantity, and representativeness of labelled training examples.
- Combining two model scores does not automatically improve classification accuracy.
- Satellite observation timing, spatial resolution, cloud cover, and other observation conditions can affect the available evidence.
- The system should not be treated as a replacement for official fire alerts, field verification, or emergency response systems.

## 🚀 Future Scope

Potential improvements to WildfireWatch include:

- Integrating matched VIIRS radiance and geolocation imagery into the image-processing pipeline.
- Expanding the dataset to include independently verified wildfire and industrial heat events.
- Evaluating performance across different regions, seasons, and environmental conditions.
- Comparing PCA-only, CNN-only, and PCA + CNN ensemble performance.
- Calibrating model probabilities and selecting decision thresholds using validation data.
- Investigating additional contextual information, such as vegetation, land cover, and weather conditions.
- Developing a more comprehensive evaluation framework for false positives and missed detections.
- Exploring near-real-time analysis when suitable satellite observations become available.

## 📚 References

1. [NASA FIRMS — Fire Information for Resource Management System](https://firms.modaps.eosdis.nasa.gov/)
2. [NASA LAADS DAAC — Level-1 and Atmosphere Archive & Distribution System](https://ladsweb.modaps.eosdis.nasa.gov/)

---

**WildfireWatch** — Exploring satellite-based thermal anomaly classification through PCA, CNNs, and geographical context.
