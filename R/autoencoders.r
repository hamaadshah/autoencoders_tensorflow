# Author: Hamaad Shah.
install.packages(pkgs = "https://cran.r-project.org/bin/macosx/el-capitan/contrib/3.4/insuranceData_1.0.tgz",
                 lib = .libPaths()[1],
                 repos = NULL)

libs.to.load <- list("insuranceData",
                     "data.table",
                     "hmeasure")

invisible(lapply(X = libs.to.load,
                 FUN = function(x) library(package = x,
                                           lib.loc = .libPaths()[1],
                                           character.only = TRUE)))

data(ClaimsLong)
ins.data.dt <- data.table(ClaimsLong)
# Although the ClaimsLong data is in the right structure that I am looking for...
# ...it is not quite interesting enough...
# ...the only value that appears to change over policy years is the number of claims and whether a claim was made or not...
# ...where ideally I would've liked some historical transactions data per policy-period combination...
# ...so I will try to create a simple dataset myself.

set.seed(seed = 666)
n.policies <- 1000
n.time.periods <- 4
train.prop <- 0.7

claim.risk <- data.table("policy.id" = 1:n.policies,
                         "claim.risk" = rbinom(n = n.policies, size = 1, prob = 0.25))
write.csv(x = claim.risk, file = "R/data/claim_risk.csv", row.names = FALSE)

transactions.generator <- function(claim.risk) {
  if (claim.risk == 0) {
    return(data.table("payments" = rbinom(n = n.time.periods, size = 1, prob = 0.25) * rgamma(n = n.time.periods, shape = 1, rate = 0.1),
                      "insured.value" = rgamma(n = n.time.periods, shape = 1, rate = 0.01),
                      "premium" = rgamma(n = n.time.periods, shape = 1, rate = 0.05)))
  } else {
    return(data.table("payments" = rbinom(n = n.time.periods, size = 1, prob = 0.5) * rgamma(n = n.time.periods, shape = 1, rate = 0.1),
                      "insured.value" = rgamma(n = n.time.periods, shape = 1, rate = 0.005),
                      "premium" = rgamma(n = n.time.periods, shape = 1, rate = 0.025)))
  }
}

transactions <- lapply(X = 1:n.policies,
                       FUN = function(policy.id) cbind(policy.id,
                                                       transactions.generator(claim.risk = claim.risk$claim.risk[policy.id])))
transactions <- do.call(rbind, transactions)
write.csv(x = transactions, file = "R/data/transactions.csv", row.names = FALSE)

handcrafted.features <- transactions[, 
                                     list("feat.sum.payments" = sum(payments),
                                          "feat.median.payments" = median(payments),
                                          "feat.max.payments" = max(payments),
                                          "feat.var.payments" = var(payments),
                                          
                                          "feat.sum.insured.value" = sum(insured.value),
                                          "feat.median.insured.value" = median(insured.value),
                                          "feat.max.insured.value" = max(insured.value),
                                          "feat.var.insured.value" = var(insured.value),
                                          
                                          "feat.sum.premium" = sum(premium),
                                          "feat.median.premium" = median(premium),
                                          "feat.max.premium" = max(premium),
                                          "feat.var.premium" = var(premium)), 
                                     by = "policy.id"][, policy.id := NULL]
write.csv(x = handcrafted.features, file = "R/data/handcrafted_features.csv", row.names = FALSE)

tr.ind <- sample(x = 1:n.policies, size = floor(train.prop * n.policies), replace = FALSE)
ts.ind <- setdiff(x = 1:n.policies, y = tr.ind)

mu <- apply(X = handcrafted.features[tr.ind], MARGIN = 2, FUN = mean)
sd.dev <- apply(X = handcrafted.features[tr.ind], MARGIN = 2, FUN = sd)

handcrafted.features.z <- sweep(x = as.matrix(handcrafted.features), MARGIN = 2, STATS = mu, FUN = "-")
handcrafted.features.z <- sweep(x = handcrafted.features.z, MARGIN = 2, STATS = sd.dev, FUN = "/")

glm.mod <- glm(formula = claim.risk ~ ., 
               family = binomial("logit"),
               data = data.table("claim.risk" = claim.risk$claim.risk, handcrafted.features.z)[tr.ind],
               control = list(maxit = 10000))

summary(HMeasure(true.class = claim.risk$claim.risk[ts.ind],
                 scores = predict(object = glm.mod, newdata = data.table(handcrafted.features.z)[ts.ind], type = "response")))