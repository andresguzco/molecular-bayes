from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

net = EquiformerV2_OC20(10, 10, 10)
print(type(m) for m in net.modules())