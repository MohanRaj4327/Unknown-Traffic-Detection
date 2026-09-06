package com.ccsutd.miniproject.dto;

import weka.core.Instances;

/**
 * Data Transfer Object holding the separated unlabelled data.
 */
public class OcSvmResult {

    private Instances pseudoNegatives; // "M" - Candidates rejected by OC-SVMs with high CfDmax
    private Instances likelyKnowns;    // "M2" - Accepted by at least one OC-SVM

    public OcSvmResult(Instances pseudoNegatives, Instances likelyKnowns) {
        this.pseudoNegatives = pseudoNegatives;
        this.likelyKnowns = likelyKnowns;
    }

    public Instances getPseudoNegatives() {
        return pseudoNegatives;
    }

    public Instances getLikelyKnowns() {
        return likelyKnowns;
    }
}
