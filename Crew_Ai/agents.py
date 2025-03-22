import numpy as np

class ContinuityAgent:
    def monitor(self, voltage, current):
        status = "✅ Stable Continuity"
        if current < 5:
            status = "⚠️ Low Current: Possible Power Cut"
        elif voltage < 200:
            status = "⚠️ Voltage Drop: Risk of Interruption"

        return {"voltage": voltage, "current": current, "status": status}

class StabilityAgent:
    def monitor(self, load):
        status = "✅ Stable Load"
        if load > 80:
            status = "⚠️ High Load: Risk of Overload"
        elif load < 20:
            status = "⚠️ Low Load: Inefficient Usage"

        return {"load": load, "status": status}

class QualityAgent:
    def monitor(self, voltage, harmonics):
        status = "✅ Normal Quality"
        if voltage < 210 or voltage > 230:
            status = "⚠️ Voltage Issue"
        if harmonics > 5:
            status += " | 🔴 High Harmonics"

        return {"voltage": voltage, "harmonics": harmonics, "status": status}
