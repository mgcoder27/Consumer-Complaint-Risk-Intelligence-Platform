-- Create the CFPB complaints star schema in the selected MySQL database.
-- Run this file before running src/load_complaints.py.

CREATE TABLE IF NOT EXISTS dim_company (
    company_id INT UNSIGNED NOT NULL AUTO_INCREMENT,
    company_name VARCHAR(512) NOT NULL,
    PRIMARY KEY (company_id),
    UNIQUE KEY uq_dim_company_name (company_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS dim_product_issue (
    -- The requested schema has one dimension for both product and issue.
    -- These source labels form a composite natural key for each combination.
    product_id VARCHAR(255) NOT NULL,
    issue_id VARCHAR(512) NOT NULL,
    PRIMARY KEY (product_id, issue_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS dim_state (
    state_id SMALLINT UNSIGNED NOT NULL AUTO_INCREMENT,
    state_code VARCHAR(10) NOT NULL,
    PRIMARY KEY (state_id),
    UNIQUE KEY uq_dim_state_code (state_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS dim_date (
    date_id INT UNSIGNED NOT NULL,
    date_value DATE NOT NULL,
    year SMALLINT UNSIGNED NOT NULL,
    month TINYINT UNSIGNED NOT NULL,
    month_name VARCHAR(9) NOT NULL,
    quarter TINYINT UNSIGNED NOT NULL,
    PRIMARY KEY (date_id),
    UNIQUE KEY uq_dim_date_value (date_value)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS fact_complaints (
    complaint_id BIGINT UNSIGNED NOT NULL,
    date_received_id INT UNSIGNED NOT NULL,
    company_id INT UNSIGNED NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    issue_id VARCHAR(512) NOT NULL,
    state_id SMALLINT UNSIGNED NOT NULL,
    narrative TEXT NULL,
    company_response VARCHAR(255) NULL,
    timely_response TINYINT(1) NULL,
    consumer_disputed TINYINT(1) NULL,
    PRIMARY KEY (complaint_id),
    KEY ix_fact_date_received (date_received_id),
    KEY ix_fact_company (company_id),
    KEY ix_fact_state (state_id),
    KEY ix_fact_product_issue (product_id, issue_id),
    CONSTRAINT fk_fact_date
        FOREIGN KEY (date_received_id) REFERENCES dim_date (date_id),
    CONSTRAINT fk_fact_company
        FOREIGN KEY (company_id) REFERENCES dim_company (company_id),
    CONSTRAINT fk_fact_product_issue
        FOREIGN KEY (product_id, issue_id)
        REFERENCES dim_product_issue (product_id, issue_id),
    CONSTRAINT fk_fact_state
        FOREIGN KEY (state_id) REFERENCES dim_state (state_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
