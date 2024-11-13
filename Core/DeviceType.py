from copy import deepcopy
from collections import OrderedDict

DeviceType = OrderedDict([
    ('PMOS', {
        'pmos': ['Drain', 'Source', 'Gate'],
        'pmos-cross': ['Drain', 'Source', 'Gate'],
        'pmos-bulk': ['Drain', 'Source', 'Gate', 'Body'],
        'default': ['Drain', 'Source', 'Gate', 'Body']
    }),
    ('NMOS', {
        'nmos': ['Drain', 'Source', 'Gate'],
        'nmos-cross': ['Drain', 'Source', 'Gate'],
        'nmos-bulk': ['Drain', 'Source', 'Gate', 'Body'],
        'default': ['Drain', 'Source', 'Gate', 'Body']
    }),
    ('Voltage', {
        'Voltage_1': ['Positive', 'Negative'],
        'Voltage_2': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('Current', {
        'current': ['Out', 'In'],
        'default': ['Out', 'In']
    }),
    ('NPN', {
        'npn': ['Base', 'Emitter', 'Collector'],
        'npn-cross': ['Base', 'Emitter', 'Collector'],
        'default': ['Base', 'Emitter', 'Collector']
    }),
    ('PNP', {
        'pnp': ['Base', 'Emitter', 'Collector'],
        'pnp-cross': ['Base', 'Emitter', 'Collector'],
        'default': ['Base', 'Emitter', 'Collector']
    }),
    ('Diode', {
        'diode': ['In', 'Out'],
        'default': ['In', 'Out']
    }),
    ('Diso_amp', {
        'Diso_amp': ['InN', 'InP', 'Out'],
        'default': ['InN', 'InP', 'Out']
    }),
    ('Siso_amp', {
        'Siso_amp': ['In', 'Out'],
        'default': ['In', 'Out']
    }),
    ('Dido_amp', {
        'Dido_amp': ['InN', 'InP', 'OutN', 'OutP'],
        'default': ['InN', 'InP', 'OutN', 'OutP']
    }),
    ('Cap', {
        'capacitor': ['Pos', 'Neg'],
        'default': ['Pos', 'Neg']
    }),
    ('Gnd', {
        'gnd': ['port'],
        'default': ['port']
    }),
    ('Ind', {
        'inductor': ['Pos', 'Neg'],
        'default': ['Pos', 'Neg']
    }),
    ('Res', {
        'resistor': ['Pos', 'Neg'],
        'resistor2': ['Pos', 'Neg'],
        'default': ['Pos', 'Neg']
    }),
    ('VDD', {
        'vdd': ['port'],
        'default': ['port']
    }),
    ('BasicCircuit', {
        'cross': [],
        'corner': [],
        'switch': [],
        'switch-3': [],
        'default': []
    })
])

DeviceTypeYolo = OrderedDict([
    ('PMOS', {
        'pmos': ['Drain', 'Source', 'Gate'],
        'pmos-cross': ['Drain', 'Source', 'Gate'],
        'pmos-bulk': ['Drain', 'Source', 'Gate', 'Body'],
        'default': ['Drain', 'Source', 'Gate', 'Body']
    }),
    ('NMOS', {
        'nmos': ['Drain', 'Source', 'Gate'],
        'nmos-cross': ['Drain', 'Source', 'Gate'],
        'nmos-bulk': ['Drain', 'Source', 'Gate', 'Body'],
        'default': ['Drain', 'Source', 'Gate', 'Body']
    }),
    ('Voltage', {
        'Voltage_1': ['Positive', 'Negative'],
        'Voltage_2': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('Current', {
        'current': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('NPN', {
        'npn': ['Base', 'Emitter', 'Collect'],
        'npn-cross': ['Base', 'Emitter', 'Collect'],
        'default': ['Base', 'Emitter', 'Collect']
    }),
    ('PNP', {
        'pnp': ['Base', 'Emitter', 'Collect'],
        'pnp-cross': ['Base', 'Emitter', 'Collect'],
        'default': ['Base', 'Emitter', 'Collect']
    }),
    ('Diode', {
        'diode': ['In', 'Out'],
        'default': ['In', 'Out']
    }),
    ('Diso_amp', {
        'Diso_amp': ['InN', 'InP', 'Out'],
        'default': ['InN', 'InP', 'Out']
    }),
    ('Siso_amp', {
        'Siso_amp': ['In', 'Out'],
        'default': ['In', 'Out']
    }),
    ('Dido_amp', {
        'Dido_amp': ['InN', 'InP', 'OutN', 'OutP'],
        'default': ['InN', 'InP', 'OutN', 'OutP']
    }),
    ('Cap', {
        'capacitor': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('Gnd', {
        'gnd': ['port'],
        'default': ['port']
    }),
    ('Ind', {
        'inductor': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('Res', {
        'resistor': ['Positive', 'Negative'],
        'resistor2': ['Positive', 'Negative'],
        'default': ['Positive', 'Negative']
    }),
    ('VDD', {
        'vdd': ['port'],
        'default': ['port']
    }),
    ('BasicCircuit', {
        'cross': [],
        'corner': [],
        'switch': [],
        'switch-3': [],
        'default': []
    })
])

# print(DeviceType)

swappedDeviceType = {}

for primary_key, secondary_dict in DeviceType.items():
    for secondary_key in secondary_dict.keys():
        if secondary_key == 'default':
            continue
        swappedDeviceType[secondary_key] = primary_key

# print(swappedDeviceType)

key_indices = {key: index + 1 for index, key in enumerate(DeviceType.keys())}
# print(key_indices)

DeviceTypeSecondary = {}
for i in DeviceType.values():
    for key, value in i.items():
        if key != 'default':
            DeviceTypeSecondary[key] = value

# print(DeviceTypeSecondary)

CircuitType = ['DISO-Amplifier', 'DIDO-Amplifier', 'SISO-Amplifier', 'Bandgap', 'LDO', 'Comparator']


def process():
    DeviceTypeTrainYolo = deepcopy(DeviceTypeYolo)
    # DeviceTypeTrainYolo['BasicCircuit'].pop('corner')
    for key in DeviceTypeTrainYolo.keys():
        DeviceTypeTrainYolo[key].pop('default')
    # # DeviceTypeTrainYolo['Voltage']['Voltage_1'] = []
    # # DeviceTypeTrainYolo['Voltage']['Voltage_2'] = []
    # DeviceTypeTrainYolo['Cap']['capacitor'] = []
    # DeviceTypeTrainYolo['Ind']['inductor'] = []
    # DeviceTypeTrainYolo['Res']['resistor'] = []
    # DeviceTypeTrainYolo['Res']['resistor2'] = []

    # yolo yaml point & obj detection

    deviceTypes = []
    for i in DeviceTypeTrainYolo.values():
        deviceTypes += list(i.keys())
    with open('../model/yolo/model.txt', 'w') as f:
        f.write("\n".join(deviceTypes))
    f.close()
    print(deviceTypes)
    print(len(deviceTypes))

    import yaml
    from collections import OrderedDict

    ordered_deviceTypes = OrderedDict()
    ordered_deviceTypes['has_visible'] = 'true'
    ordered_deviceTypes['classes'] = OrderedDict()
    for device_category, types in DeviceTypeTrainYolo.items():
        for type, position in types.items():
            ordered_deviceTypes['classes'][type] = position

    def represent_ordereddict(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())

    yaml.add_representer(OrderedDict, represent_ordereddict)
    yaml_data = yaml.dump(ordered_deviceTypes, allow_unicode=True)
    print(yaml_data)

    with open('../model/yolo/model.yaml', 'w') as f:
        f.write(yaml_data)

    print('\n'.join(CircuitType))


if __name__ == '__main__':
    # ...
    process()
