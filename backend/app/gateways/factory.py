import os
from backend.app.gateways.base import PaymentGateway
from backend.app.gateways.simulator import SimulatorGateway
from backend.app.gateways.razorpay import RazorpayGatewayAdapter

class GatewayFactory:
    @staticmethod
    def get_gateway(gateway_type: str = "SIMULATOR") -> PaymentGateway:
        selected_type = os.getenv("PAYMENT_GATEWAY_TYPE", gateway_type).upper()

        if selected_type in ("RAZORPAY", "PROD"):
            return RazorpayGatewayAdapter()
        return SimulatorGateway()

_simulator_gateway_instance = None

def get_simulator_gateway() -> SimulatorGateway:
    global _simulator_gateway_instance
    if _simulator_gateway_instance is None:
        _simulator_gateway_instance = SimulatorGateway()
    return _simulator_gateway_instance

def get_default_gateway() -> PaymentGateway:
    return GatewayFactory.get_gateway()
