package com.ccsutd.miniproject.service;

import org.springframework.stereotype.Service;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.io.IOException;

@Service
public class DatasetService {

    /**
     * Loads an ARFF file using Weka's ArffLoader.
     * 
     * @param filePath the path to the ARFF dataset file
     * @return Instances object representing the loaded dataset
     * @throws IOException if the file cannot be found or read
     */
    public Instances loadDataset(String filePath) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File(filePath));
        Instances data = loader.getDataSet();
        
        // Ensure the last attribute is set as the class attribute
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        
        return data;
    }
}
