package com.ccsutd.miniproject.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "experiment_runs")
public class ExperimentRun {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private LocalDateTime runDate;
    
    private Double atsCalculatedAlpha;
    private Integer pseudoNegativesUsed;
    private Integer likelyKnownsUsed;
    private Long h1TrainTimeMs;
    
    private Integer totalTested;
    private Integer correctKnown;
    private Integer correctNew;
    private Integer falseKnown;
    private Integer falseNew;
    private Integer h1EarlyBlocks;
    
    private Double knownAccuracy;
    private Double unknownAccuracy;
    private Double normalizedAccuracy;

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public LocalDateTime getRunDate() { return runDate; }
    public void setRunDate(LocalDateTime runDate) { this.runDate = runDate; }

    public Double getAtsCalculatedAlpha() { return atsCalculatedAlpha; }
    public void setAtsCalculatedAlpha(Double atsCalculatedAlpha) { this.atsCalculatedAlpha = atsCalculatedAlpha; }

    public Integer getPseudoNegativesUsed() { return pseudoNegativesUsed; }
    public void setPseudoNegativesUsed(Integer pseudoNegativesUsed) { this.pseudoNegativesUsed = pseudoNegativesUsed; }

    public Integer getLikelyKnownsUsed() { return likelyKnownsUsed; }
    public void setLikelyKnownsUsed(Integer likelyKnownsUsed) { this.likelyKnownsUsed = likelyKnownsUsed; }

    public Long getH1TrainTimeMs() { return h1TrainTimeMs; }
    public void setH1TrainTimeMs(Long h1TrainTimeMs) { this.h1TrainTimeMs = h1TrainTimeMs; }

    public Integer getTotalTested() { return totalTested; }
    public void setTotalTested(Integer totalTested) { this.totalTested = totalTested; }

    public Integer getCorrectKnown() { return correctKnown; }
    public void setCorrectKnown(Integer correctKnown) { this.correctKnown = correctKnown; }

    public Integer getCorrectNew() { return correctNew; }
    public void setCorrectNew(Integer correctNew) { this.correctNew = correctNew; }

    public Integer getFalseKnown() { return falseKnown; }
    public void setFalseKnown(Integer falseKnown) { this.falseKnown = falseKnown; }

    public Integer getFalseNew() { return falseNew; }
    public void setFalseNew(Integer falseNew) { this.falseNew = falseNew; }

    public Integer getH1EarlyBlocks() { return h1EarlyBlocks; }
    public void setH1EarlyBlocks(Integer h1EarlyBlocks) { this.h1EarlyBlocks = h1EarlyBlocks; }

    public Double getKnownAccuracy() { return knownAccuracy; }
    public void setKnownAccuracy(Double knownAccuracy) { this.knownAccuracy = knownAccuracy; }

    public Double getUnknownAccuracy() { return unknownAccuracy; }
    public void setUnknownAccuracy(Double unknownAccuracy) { this.unknownAccuracy = unknownAccuracy; }

    public Double getNormalizedAccuracy() { return normalizedAccuracy; }
    public void setNormalizedAccuracy(Double normalizedAccuracy) { this.normalizedAccuracy = normalizedAccuracy; }
}
