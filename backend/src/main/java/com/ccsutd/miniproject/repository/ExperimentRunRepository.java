package com.ccsutd.miniproject.repository;

import com.ccsutd.miniproject.entity.ExperimentRun;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ExperimentRunRepository extends JpaRepository<ExperimentRun, Long> {
}
