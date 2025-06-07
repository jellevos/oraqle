use std::time::Instant;

use tfhe::{generate_keys, prelude::*, set_server_key, CompressedServerKey, ConfigBuilder, FheBool};
//use tfhe::shortint::{gen_keys, CompressedServerKey};
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::safe_serialization::{safe_deserialize_conformant, safe_serialize};
use tfhe::FheUint8;

fn main() {
    let config = ConfigBuilder::default().build();
    let (client_key, server_key) = generate_keys(config);

    let compressed = CompressedServerKey::new(&client_key);
    let serialized = bincode::serialize(&compressed).unwrap();
    println!("{}", serialized.len());

    let clear_b = 50u8;

    set_server_key(server_key);

    let man = FheBool::encrypt(true, &client_key);
    let woman = FheBool::encrypt(true, &client_key);
    let age = FheUint8::encrypt(clear_b, &client_key);
    let smoking = FheBool::encrypt(true, &client_key);
    let diabetic = FheBool::encrypt(true, &client_key);
    let hbp = FheBool::encrypt(true, &client_key);
    let cholesterol = FheUint8::encrypt(clear_b, &client_key);
    let weight = FheUint8::encrypt(clear_b, &client_key);
    let height = FheUint8::encrypt(clear_b, &client_key);
    let activity = FheUint8::encrypt(clear_b, &client_key);
    let alcohol = FheUint8::encrypt(clear_b, &client_key);

    let start = Instant::now();

    let mut all = vec![];
    for _ in 0..10 {
        let mut res: FheUint8 = (&man & age.gt(50)).cast_into();

        let cond: FheUint8 = (&woman & age.gt(60)).cast_into();
        res += cond;

        let cond: FheUint8 = smoking.clone().cast_into();
        res += cond;

        let cond: FheUint8 = diabetic.clone().cast_into();
        res += cond;

        let cond: FheUint8 = hbp.clone().cast_into();
        res += cond;

        let cond: FheUint8 = (cholesterol.lt(40)).cast_into();
        res += cond;

        let cond: FheUint8 = (weight.gt(&height - 90)).cast_into();
        res += cond;

        let cond: FheUint8 = (activity.lt(30)).cast_into();
        res += cond;

        let cond: FheUint8 = (&man & alcohol.gt(3)).cast_into();
        res += cond;

        let cond: FheUint8 = (&woman & alcohol.gt(2)).cast_into();
        res += cond;

        all.push(res);
    }

    println!("{:?}", start.elapsed() / 10);

    for res in all {
        let decrypted: u64 = res.decrypt(&client_key);
        println!("{}", decrypted);
    }



    let start = Instant::now();

    let mut all = vec![];
    for _ in 0..10 {
        let mut res: FheBool = &man & age.gt(50);

        let cond: FheBool = &woman & age.gt(60);
        res &= cond;

        let cond: &FheBool = &smoking;
        res &= cond;

        let cond: &FheBool = &diabetic;
        res &= cond;

        let cond: &FheBool = &hbp;
        res &= cond;

        let cond: FheBool = cholesterol.lt(40);
        res &= cond;

        let cond: FheBool = weight.gt(&height - 90);
        res &= cond;

        let cond: FheBool = activity.lt(30);
        res &= cond;

        let cond: FheBool = &man & alcohol.gt(3);
        res &= cond;

        let cond: FheBool = &woman & alcohol.gt(2);
        res &= cond;

        all.push(res);
    }

    println!("{:?}", start.elapsed() / 10);

    for res in all {
        let decrypted: bool = res.decrypt(&client_key);
        println!("{}", decrypted);
    }
}
