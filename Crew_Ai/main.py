import time
import random
from agents import ContinuityAgent, StabilityAgent, QualityAgent
from database import session, ElectricData

# ✅ Initialize Agents
continuity_agent = ContinuityAgent()
stability_agent = StabilityAgent()
quality_agent = QualityAgent()

# ✅ Function to Simulate Real-Time Data
def get_real_time_data():
    return {
        "voltage": random.uniform(200, 240),  # Simulate voltage fluctuations
        "current": random.uniform(5, 15),  # Simulate current changes
        "harmonics": random.uniform(0, 10),  # Simulate harmonic variations
        "load": random.uniform(10, 90)  # Simulate load variation
    }

# ✅ Store Data in PostgreSQL
def store_results(results):
    new_entry = ElectricData(
        voltage=results.get("voltage", 0),
        current=results.get("current", 0),
        harmonics=results.get("harmonics", 0),
        load=results.get("load", 0),
        status=results["status"]
    )
    session.add(new_entry)
    session.commit()

# ✅ Real-Time Monitoring Loop
try:
    while True:
        # Get new sensor data
        data = get_real_time_data()
        
        # Process data through agents
        continuity_result = continuity_agent.monitor(data["voltage"], data["current"])
        stability_result = stability_agent.monitor(data["load"])
        quality_result = quality_agent.monitor(data["voltage"], data["harmonics"])

        # Store results in PostgreSQL
        store_results(continuity_result)
        store_results(stability_result)
        store_results(quality_result)

        print("✅ Real-time data stored in PostgreSQL:", data)
        
        time.sleep(5)  # Wait 5 seconds before next reading

except KeyboardInterrupt:
    print("\n🛑 Real-time monitoring stopped.")
