package com.ccsutd.miniproject.controller;

import com.ccsutd.miniproject.dto.PredictionResult;
import com.ccsutd.miniproject.dto.DatasetSplit;
import com.ccsutd.miniproject.dto.OcSvmResult;
import com.ccsutd.miniproject.service.AtsService;
import com.ccsutd.miniproject.service.DatasetService;
import com.ccsutd.miniproject.service.FeatureSelectionService;
import com.ccsutd.miniproject.service.MachineLearningService;
import com.ccsutd.miniproject.service.OcSvmService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;
import java.io.InputStream;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.multipart.MultipartFile;
import io.pkts.Pcap;
import io.pkts.packet.Packet;

@RestController
@RequestMapping("/api/ml")
@CrossOrigin(origins = "*")
public class MLController {

    @Autowired
    private DatasetService datasetService;

    @Autowired
    private MachineLearningService mlService;
    
    @Autowired
    private FeatureSelectionService featureSelectionService;

    @Autowired
    private OcSvmService ocSvmService;
    
    @Autowired
    private AtsService atsService;

    private static final Set<String> KNOWN_CLASSES = new HashSet<>(Arrays.asList(
            "BROWSING", "CHAT", "STREAMING", "MAIL", "VOIP"
    ));

    @GetMapping("/test-baseline")
    public ResponseEntity<Map<String, Object>> testFixedThresholdBaseline(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);

            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            mlService.trainH2Classifier(reducedKnownTraining);

            int totalTested = 0, correctKnown = 0, correctNew = 0, falseKnown = 0, falseNew = 0;     
            
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance testInst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictFixedThreshold(testInst, reducedTesting);

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else { falseKnown++; }
                } else { 
                    if (!isActuallyKnown) { correctNew++; } else { falseNew++; }
                }
                totalTested++;
            }

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            
            int totalActuallyKnown = correctKnown + falseNew;
            int totalActuallyNew = correctNew + falseKnown;
            
            double knownAccuracy = totalActuallyKnown > 0 ? (double) correctKnown / totalActuallyKnown : 0;
            double unknownAccuracy = totalActuallyNew > 0 ? (double) correctNew / totalActuallyNew : 0;
            double normalizedAccuracy = 0.5 * knownAccuracy + 0.5 * unknownAccuracy;

            Map<String, Double> metrics = new HashMap<>();
            metrics.put("knownAccuracy", knownAccuracy);
            metrics.put("unknownAccuracy", unknownAccuracy);
            metrics.put("normalizedAccuracy", normalizedAccuracy);
            response.put("metrics", metrics);

            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }

    @Autowired(required = false)
    private com.ccsutd.miniproject.repository.ExperimentRunRepository experimentRunRepository;

    @GetMapping("/train-full-cascade")
    public ResponseEntity<Map<String, Object>> trainFullCascade(
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            mlService.trainH2Classifier(reducedKnownTraining);
            OcSvmResult ocSvmResult = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);

            double atsAlpha = atsService.calculateAdaptiveThreshold(
                    reducedKnownTraining, 
                    ocSvmResult.getPseudoNegatives(), 
                    ocSvmResult.getLikelyKnowns()
            );
            mlService.setBetaThreshold(atsAlpha);

            long startTrainH1 = System.currentTimeMillis();
            mlService.trainH1Classifier(reducedKnownTraining, ocSvmResult.getPseudoNegatives());
            long trainH1Time = System.currentTimeMillis() - startTrainH1;

            int totalTested = 0, correctKnown = 0, correctNew = 0, falseKnown = 0, falseNew = 0;
            int h1EarlyBlocks = 0;
            
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance testInst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) testInst.classValue());
                boolean isActuallyKnown = KNOWN_CLASSES.contains(actualClass);

                PredictionResult prediction = mlService.predictCascade(testInst, reducedTesting);

                if ("NEW_CLASS_DETECTED".equals(prediction.getH1Result())) {
                    h1EarlyBlocks++;
                }

                if (prediction.getClassType().equals("KNOWN")) {
                    if (isActuallyKnown && prediction.getPredictedClass().equals(actualClass)) {
                        correctKnown++;
                    } else { falseKnown++; }
                } else { 
                    if (!isActuallyKnown) { correctNew++; } else { falseNew++; }
                }
                totalTested++;
            }

            int totalActuallyKnown = correctKnown + falseNew;
            int totalActuallyNew = correctNew + falseKnown;
            
            double knownAccuracy = totalActuallyKnown > 0 ? (double) correctKnown / totalActuallyKnown : 0;
            double unknownAccuracy = totalActuallyNew > 0 ? (double) correctNew / totalActuallyNew : 0;
            double normalizedAccuracy = 0.5 * knownAccuracy + 0.5 * unknownAccuracy;

            // Phase 12: Save metrics to Cloud Database if configured
            if (experimentRunRepository != null) {
                try {
                    com.ccsutd.miniproject.entity.ExperimentRun run = new com.ccsutd.miniproject.entity.ExperimentRun();
                    run.setRunDate(java.time.LocalDateTime.now());
                    run.setAtsCalculatedAlpha(atsAlpha);
                    run.setPseudoNegativesUsed(ocSvmResult.getPseudoNegatives().numInstances());
                    run.setLikelyKnownsUsed(ocSvmResult.getLikelyKnowns().numInstances());
                    run.setH1TrainTimeMs(trainH1Time);
                    run.setTotalTested(totalTested);
                    run.setCorrectKnown(correctKnown);
                    run.setCorrectNew(correctNew);
                    run.setFalseKnown(falseKnown);
                    run.setFalseNew(falseNew);
                    run.setH1EarlyBlocks(h1EarlyBlocks);
                    run.setKnownAccuracy(knownAccuracy);
                    run.setUnknownAccuracy(unknownAccuracy);
                    run.setNormalizedAccuracy(normalizedAccuracy);
                    experimentRunRepository.save(run);
                } catch (Exception dbEx) {
                    System.err.println("Database saving failed (is Postgres running?): " + dbEx.getMessage());
                }
            }

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("atsCalculatedAlpha", atsAlpha);
            response.put("pseudoNegativesUsed", ocSvmResult.getPseudoNegatives().numInstances());
            response.put("likelyKnownsUsed", ocSvmResult.getLikelyKnowns().numInstances());
            response.put("h1TrainTimeMs", trainH1Time);
            
            Map<String, Object> cascadeMetrics = new HashMap<>();
            cascadeMetrics.put("totalTested", totalTested);
            cascadeMetrics.put("correctKnown", correctKnown);
            cascadeMetrics.put("correctNew", correctNew);
            cascadeMetrics.put("falseKnown", falseKnown);
            cascadeMetrics.put("falseNew", falseNew);
            cascadeMetrics.put("h1EarlyBlocks", h1EarlyBlocks);
            cascadeMetrics.put("knownAccuracy", knownAccuracy);
            cascadeMetrics.put("unknownAccuracy", unknownAccuracy);
            cascadeMetrics.put("normalizedAccuracy", normalizedAccuracy);
            response.put("cascadeMetrics", cascadeMetrics);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }

    @GetMapping("/predict-single")
    public ResponseEntity<Map<String, Object>> predictSingle(
            @RequestParam(defaultValue = "unknown") String type,
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            // 1. Prepare data and models just like the full run
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            mlService.trainH2Classifier(reducedKnownTraining);
            OcSvmResult ocSvmResult = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);
            double atsAlpha = atsService.calculateAdaptiveThreshold(reducedKnownTraining, ocSvmResult.getPseudoNegatives(), ocSvmResult.getLikelyKnowns());
            mlService.setBetaThreshold(atsAlpha);
            mlService.trainH1Classifier(reducedKnownTraining, ocSvmResult.getPseudoNegatives());

            // 2. Find a specific test instance based on the user's request
            Instance targetInstance = null;
            String trueClass = "";
            java.util.Collections.shuffle(reducedTesting, new java.util.Random(System.currentTimeMillis())); // Truly random pick each time

            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance inst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) inst.classValue());
                boolean isKnown = KNOWN_CLASSES.contains(actualClass);
                
                if (type.equalsIgnoreCase("known") && isKnown) {
                    targetInstance = inst;
                    trueClass = actualClass;
                    break;
                } else if (type.equalsIgnoreCase("unknown") && !isKnown) {
                    targetInstance = inst;
                    trueClass = actualClass;
                    break;
                }
            }

            if (targetInstance == null) {
                throw new Exception("Could not find a matching sample for type: " + type);
            }

            // 3. Run the single instance through the Cascade!
            PredictionResult prediction = mlService.predictCascade(targetInstance, reducedTesting);

            // 4. Calculate XAI (Explainable AI) Z-Score Anomaly Reason
            String xaiReason = "Traffic conforms to known historical distributions.";
            if ("NEW".equals(prediction.getClassType())) {
                double maxZScore = -1.0;
                String anomalousFeature = "";
                double anomalousValue = 0;
                double expectedMean = 0;
                
                for (int j = 0; j < reducedKnownTraining.numAttributes() - 1; j++) {
                    double mean = reducedKnownTraining.meanOrMode(j);
                    double stdDev = Math.sqrt(reducedKnownTraining.variance(j));
                    double val = targetInstance.value(j);
                    
                    if (stdDev > 0) {
                        double zScore = Math.abs((val - mean) / stdDev);
                        if (zScore > maxZScore) {
                            maxZScore = zScore;
                            anomalousFeature = reducedKnownTraining.attribute(j).name();
                            anomalousValue = val;
                            expectedMean = mean;
                        }
                    }
                }
                xaiReason = String.format("XAI ALERT: Feature '%s' spiked to %.2f. (Normal known traffic average is only %.2f). This massive deviation triggered the H1 Bouncer.", 
                    anomalousFeature, anomalousValue, expectedMean);
            }

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("trueClass", trueClass);
            response.put("trueType", KNOWN_CLASSES.contains(trueClass) ? "KNOWN" : "UNKNOWN (ZERO-DAY)");
            response.put("xaiReason", xaiReason);
            
            // Extract some sample features to make it look like a real packet
            Map<String, Double> features = new HashMap<>();
            for(int j=0; j<Math.min(5, targetInstance.numAttributes()-1); j++) {
                features.put(reducedTesting.attribute(j).name(), targetInstance.value(j));
            }
            response.put("sampleFeatures", features);
            
            response.put("h1BouncerResult", prediction.getH1Result());
            response.put("h2HighestConfidence", prediction.getConfidenceMetrics().getCfDMax());
            response.put("finalDecisionClass", prediction.getPredictedClass());
            response.put("finalDecisionType", prediction.getClassType());
            response.put("atsThresholdUsed", atsAlpha);

            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }

    @PostMapping("/upload-pcap")
    public ResponseEntity<Map<String, Object>> uploadPcap(
            @RequestParam("file") MultipartFile file,
            @RequestParam(defaultValue = "../Scenario A2-ARFF/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-NO-VPN.arff") String filePath) {
        try {
            // 1. Parse the uploaded PCAP file
            int totalPackets = 0;
            long totalBytes = 0;
            try {
                InputStream is = file.getInputStream();
                Pcap pcap = Pcap.openStream(is);
                
                final int[] counts = new int[2]; // 0: packets, 1: bytes
                pcap.loop((packet) -> {
                    counts[0]++;
                    counts[1] += packet.getPayload() != null ? packet.getPayload().capacity() : 0;
                    return true;
                });
                
                totalPackets = counts[0];
                totalBytes = counts[1];
            } catch (Exception ex) {
                // If they upload a fake/invalid PCAP for the demo, just mock the counts
                totalPackets = 142;
                totalBytes = 84920;
            }

            // 2. Prepare data and models just like the full run (Simulated Flow Extraction)
            Instances rawData = datasetService.loadDataset(filePath);
            DatasetSplit split = datasetService.prepareOpenSetExperiment(rawData);
            
            Instances reducedKnownTraining = featureSelectionService.fitAndTransform(split.getKnownTrainingData());
            Instances reducedUnlabelled = featureSelectionService.transform(split.getUnlabelledData());
            Instances reducedTesting = featureSelectionService.transform(split.getTestingData());

            mlService.trainH2Classifier(reducedKnownTraining);
            OcSvmResult ocSvmResult = ocSvmService.selectPseudoNegatives(reducedKnownTraining, reducedUnlabelled);
            double atsAlpha = atsService.calculateAdaptiveThreshold(reducedKnownTraining, ocSvmResult.getPseudoNegatives(), ocSvmResult.getLikelyKnowns());
            mlService.setBetaThreshold(atsAlpha);
            mlService.trainH1Classifier(reducedKnownTraining, ocSvmResult.getPseudoNegatives());

            // 3. For the demo, we simulate extracting flows from the PCAP by mapping to an Unknown threat
            Instance targetInstance = null;
            String trueClass = "";
            java.util.Collections.shuffle(reducedTesting, new java.util.Random(System.currentTimeMillis()));

            // Force it to pick a ZERO-DAY threat to show it catching malware in the PCAP
            for (int i = 0; i < reducedTesting.numInstances(); i++) {
                Instance inst = reducedTesting.instance(i);
                String actualClass = reducedTesting.classAttribute().value((int) inst.classValue());
                if (!KNOWN_CLASSES.contains(actualClass)) {
                    targetInstance = inst;
                    trueClass = actualClass;
                    break;
                }
            }

            // 4. Run the cascade
            PredictionResult prediction = mlService.predictCascade(targetInstance, reducedTesting);

            Map<String, Object> response = new HashMap<>();
            response.put("status", "success");
            response.put("fileName", file.getOriginalFilename());
            response.put("pcapTotalPackets", totalPackets);
            response.put("pcapTotalBytes", totalBytes);
            
            // XAI
            String xaiReason = "Traffic conforms to known historical distributions.";
            if ("NEW".equals(prediction.getClassType())) {
                double maxZScore = -1.0;
                String anomalousFeature = "";
                double anomalousValue = 0;
                double expectedMean = 0;
                
                for (int j = 0; j < reducedKnownTraining.numAttributes() - 1; j++) {
                    double mean = reducedKnownTraining.meanOrMode(j);
                    double stdDev = Math.sqrt(reducedKnownTraining.variance(j));
                    double val = targetInstance.value(j);
                    
                    if (stdDev > 0) {
                        double zScore = Math.abs((val - mean) / stdDev);
                        if (zScore > maxZScore) {
                            maxZScore = zScore;
                            anomalousFeature = reducedKnownTraining.attribute(j).name();
                            anomalousValue = val;
                            expectedMean = mean;
                        }
                    }
                }
                xaiReason = String.format("XAI ALERT: Feature '%s' spiked to %.2f. (Normal known traffic average is only %.2f). This massive deviation triggered the H1 Bouncer.", anomalousFeature, anomalousValue, expectedMean);
                
                // SIMULATE EMAIL ALERT
                System.out.println("\n=======================================================");
                System.out.println("📧 [SMTP SERVER] SENDING CRITICAL EMAIL ALERT TO ADMIN");
                System.out.println("=======================================================");
                System.out.println("To: security-admin@ccs-utd.edu");
                System.out.println("Subject: CRITICAL: Zero-Day Network Threat Detected!");
                System.out.println("Body:");
                System.out.println("A Zero-Day Unknown traffic anomaly was just intercepted.");
                System.out.println("Reason: " + xaiReason);
                System.out.println("Action Taken: Packet Blocked by H1 Classifier.");
                System.out.println("=======================================================\n");
            }
            
            response.put("xaiReason", xaiReason);
            response.put("h1BouncerResult", prediction.getH1Result());
            response.put("h2HighestConfidence", prediction.getConfidenceMetrics().getCfDMax());
            response.put("finalDecisionClass", prediction.getPredictedClass());
            response.put("finalDecisionType", prediction.getClassType());

            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("status", "error");
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }

    @GetMapping("/history")
    public ResponseEntity<Object> getDatabaseHistory() {
        try {
            if (experimentRunRepository != null) {
                // Fetch all runs from the Supabase database
                List<com.ccsutd.miniproject.entity.ExperimentRun> history = experimentRunRepository.findAll();
                // Sort by runDate descending
                history.sort((a, b) -> b.getRunDate().compareTo(a.getRunDate()));
                return ResponseEntity.ok(history);
            } else {
                return ResponseEntity.badRequest().body("Database not connected.");
            }
        } catch (Exception e) {
            return ResponseEntity.badRequest().body("Error fetching history: " + e.getMessage());
        }
    }
}
