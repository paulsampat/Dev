Initial version of  Black Scholes Prier with a simple FAST API Interface
UI where you can vary spot, K, Expiry, Vol and see the change in Price and Greeks
Next steps :
Expose as an MCP exdpoint - done
Expose as Langchain Tool

(base) paulsampat@Mac black-scholes-pricer % python langchain_agent.py "Price a call option with S=100, K=105, T=0.25, r=0.05, sigma=0.20"

[02/21/26 13:27:17] INFO     Processing request of type ListToolsRequest                                                                                                      server.py:720
/Users/paulsampat/Dev/black-scholes-pricer/langchain_agent.py:46: LangGraphDeprecatedSinceV10: create_react_agent has been moved to `langchain.agents`. Please update your import to `from langchain.agents import create_agent`. Deprecated in LangGraph V1.0 to be removed in V2.0.
  agent = create_react_agent(llm, tools)
[02/21/26 13:27:19] INFO     Processing request of type CallToolRequest                                                                                                       server.py:720
                    INFO     Processing request of type ListToolsRequest                                                                                                      server.py:720
Here's a full breakdown of the Black-Scholes pricing results for your **European Call Option**:

---

## 📋 Option Inputs
| Parameter | Value |
|---|---|
| Spot Price (S) | $100.00 |
| Strike Price (K) | $105.00 |
| Time to Expiry (T) | 0.25 years (3 months) |
| Risk-Free Rate (r) | 5.00% |
| Volatility (σ) | 20.00% |

---

## 💰 Option Price
| | |
|---|---|
| **Fair Value** | **$2.48** |

---

## 🔢 Intermediates
| | Value |
|---|---|
| d1 | -0.3129 |
| d2 | -0.4129 |

---

## 🏛️ Price Decomposition
| Component | Value |
|---|---|
| Intrinsic Value | $0.00 |
| Time Value | $2.48 |

> The option is **out-of-the-money** (S < K), so 100% of its value is **pure time value** — the market is pricing in the *probability* that the stock rises above $105 before expiry.

---

## 🇬🇷 The Greeks
| Greek | Value | Interpretation |
|---|---|---|
| **Delta** (Δ) | 0.3772 | Option price rises ~$0.38 for every $1 increase in spot |
| **Gamma** (Γ) | 0.0380 | Delta itself changes by ~0.038 for every $1 move in spot |
| **Theta** (Θ) | -$0.0256/day | Option loses ~$0.026 in value per calendar day (time decay) |
| **Vega** (ν) | $0.1899/1% vol | Option gains ~$0.19 for every 1% increase in implied vol |
| **Rho** (ρ) | $0.0881/1% rate | Option gains ~$0.088 for every 1% increase in interest rates |

---

## 🔑 Key Takeaways
- **OTM Option:** The stock needs to rally **+5%** just to reach the strike, explaining the relatively low price of $2.48.
- **Moderate Delta (0.38):** There's roughly a **38% sensitivity** to spot moves, reflecting the OTM nature.
- **Time Decay (-$0.026/day):** With only 3 months to expiry, theta erosion is a meaningful risk for the option holder.
- **Vega Sensitivity:** At $0.19 per 1% vol move, changes in implied volatility will have a noticeable impact on the option's value.
(base) paulsampat@Mac black-scholes-pricer % 

Build a Graph to represent a portfolio of options and adjust graph constituents with dynamic repricing
Add in Deep Research Module to research the latest hot investments
Build Exchange Connectivity Agent to dynamically be able to generate code for an exchange
Build an SOR using ML
Rest....
