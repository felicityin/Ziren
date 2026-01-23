use crate::memory::config::MIPS_MEMORY_SPACE;

pub const REG_FIRST_ARG: &str = "rdi";
pub const REG_SECOND_ARG: &str = "rsi";
pub const REG_THIRD_ARG: &str = "rdx";
pub const REG_FOURTH_ARG: &str = "rcx";
pub const REG_FIFTH_ARG: &str = "r8";
pub const REG_SIXTH_ARG: &str = "r9";

pub const REG_C: &str = "rdx";
pub const REG_C_W: &str = "edx";
pub const REG_C_B: &str = "dx";
pub const REG_C_LB: &str = "dl";

pub const REG_B: &str = "rsi";
pub const REG_B_W: &str = "esi";

pub const REG_A: &str = "rdi";
pub const REG_A_W: &str = "edi";

pub const REG_RETURN_VAL: &str = "rax";
pub const REG_D: &str = "rax";
pub const REG_D_W: &str = "eax";
pub const REG_INSTRET_END: &str = "r12";

pub const REG_EXEC_STATE_PTR: &str = "rbx";
pub const REG_TRACE_HEIGHT: &str = "r14";
pub const REG_AS2_PTR: &str = "r15";
pub const REG_PC: &str = "r8";
pub const REG_NEXT_PC: &str = "r9";
pub const REG_BRANCH_JMP: &str = "r7";

pub const DEFAULT_PC_OFFSET: i32 = 4;

pub const MIPS_TO_X86_OVERRIDE_MAP: [Option<&str>; 34] = [
    None,         // x0
    None,         // x1
    None,         // x2
    None,         // x3
    None,         // x4
    None,         // x5
    None,         // x6
    None,         // x7
    None,         // x8
    None,         // x9
    Some("r10d"), // x10
    Some("r11d"), // x11
    Some("r9d"),  // x12
    Some("r8d"),  // x13
    Some("ebp"),  // x14
    Some("r13d"), // x15
    None,         // x16
    None,         // x17
    None,         // x18
    None,         // x19
    None,         // x20
    None,         // x21
    None,         // x22
    None,         // x23
    None,         // x24
    None,         // x25
    None,         // x26
    None,         // x27
    None,         // x28
    None,         // x29
    None,         // x30
    None,         // x31
    None,
    None,
];

pub fn sync_xmm_to_gpr() -> String {
    let mut asm = String::new();
    for (mips_reg, override_reg_opt) in MIPS_TO_X86_OVERRIDE_MAP.iter().copied().enumerate() {
        if let Some(override_reg) = override_reg_opt {
            let xmm_reg = mips_reg / 2;
            let lane = mips_reg % 2;
            asm += &format!("   pextrd {override_reg}, xmm{xmm_reg}, {lane}\n");
        }
    }
    asm
}

pub fn sync_gpr_to_xmm() -> String {
    let mut asm = String::new();
    for (mips_reg, override_reg_opt) in MIPS_TO_X86_OVERRIDE_MAP.iter().copied().enumerate() {
        if let Some(override_reg) = override_reg_opt {
            let xmm_reg = mips_reg / 2;
            let lane = mips_reg % 2;
            asm += &format!("   pinsrd xmm{xmm_reg}, {override_reg}, {lane}\n");
        }
    }
    asm
}

/*
input:
- mips_src_reg register number
- x86_dst_gpr register to write into
- is_gpr_force_write boolean

output:
- string representing the general purpose register that stores the value of register number `mips_src_reg`
- emitted assembly string that performs the move
*/
pub fn xmm_to_gpr(
    mips_src_reg: u8,
    x86_dst_reg: &str,
    is_gpr_force_write: bool,
) -> (String, String) {
    if let Some(override_reg) = MIPS_TO_X86_OVERRIDE_MAP[mips_src_reg as usize] {
        // a is overridden, b is overridden
        if is_gpr_force_write {
            return (x86_dst_reg.to_string(), format!("  mov {x86_dst_reg}, {override_reg}\n"));
        }
        return (override_reg.to_string(), "".to_string());
    }
    if mips_src_reg < 32 {
        let xmm_map_reg = mips_src_reg / 2;
        if mips_src_reg % 2 == 0 {
            (x86_dst_reg.to_string(), format!("   pextrd {x86_dst_reg}, xmm{xmm_map_reg}, 0\n"))
        } else {
            (x86_dst_reg.to_string(), format!("   pextrd {x86_dst_reg}, xmm{xmm_map_reg}, 1\n"))
        }
    } else {
        let xmm_map_reg = (mips_src_reg - 6) / 2;
        if mips_src_reg % 2 == 0 {
            (x86_dst_reg.to_string(), format!("   pextrd {x86_dst_reg}, xmm{xmm_map_reg}, 2\n"))
        } else {
            (x86_dst_reg.to_string(), format!("   pextrd {x86_dst_reg}, xmm{xmm_map_reg}, 3\n"))
        }
    }
}

pub fn gpr_to_xmm(x86_dst_reg: &str, mips_src_reg: u8) -> String {
    if let Some(override_reg) = MIPS_TO_X86_OVERRIDE_MAP[mips_src_reg as usize] {
        if x86_dst_reg == override_reg {
            // already in correct location
            return "".to_string();
        }
        return format!("   mov {override_reg}, {x86_dst_reg}\n");
    }
    if mips_src_reg < 32 {
        let xmm_map_reg = mips_src_reg / 2;
        if mips_src_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_dst_reg}, 0\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_dst_reg}, 1\n")
        }
    } else {
        let xmm_map_reg = (mips_src_reg - 6) / 2;
        if mips_src_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_dst_reg}, 2\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_dst_reg}, 3\n")
        }
    }
}

pub fn gpr_to_mips_register(x86_reg: &str, mips_reg: u8) -> String {
    if mips_reg < 32 {
        let xmm_map_reg = mips_reg / 2;
        if mips_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_reg}, 0\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_reg}, 1\n")
        }
    } else {
        let xmm_map_reg = (mips_reg - 6) / 2;
        if mips_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_reg}, 2\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {x86_reg}, 3\n")
        }
    }
}

pub fn address_space_start_to_gpr(address_space: u32, gpr: &str) -> String {
    if address_space == MIPS_MEMORY_SPACE {
        if REG_AS2_PTR != gpr {
            return format!("    mov {gpr}, r15\n");
        }
        return "".to_string();
    }
    format!("   pextrq {gpr}, xmm0, 1\n")
}

#[derive(Copy, Clone)]
pub enum Width {
    W64,
    W32,
    W16,
    W8L,
    W8H,
}

pub fn convert_x86_reg(any: &str, to: Width) -> Option<&'static str> {
    #[rustfmt::skip]
    const T: [(&str,&str,&str,&str,Option<&str>); 16] = [
        ("rax","eax","ax","al",Some("ah")), ("rbx","ebx","bx","bl",Some("bh")),
        ("rcx","ecx","cx","cl",Some("ch")), ("rdx","edx","dx","dl",Some("dh")),
        ("rsi","esi","si","sil",None),      ("rdi","edi","di","dil",None),
        ("rbp","ebp","bp","bpl",None),      ("rsp","esp","sp","spl",None),
        ("r8","r8d","r8w","r8b",None),      ("r9","r9d","r9w","r9b",None),
        ("r10","r10d","r10w","r10b",None),  ("r11","r11d","r11w","r11b",None),
        ("r12","r12d","r12w","r12b",None),  ("r13","r13d","r13w","r13b",None),
        ("r14","r14d","r14w","r14b",None),  ("r15","r15d","r15w","r15b",None),
    ];

    fn pick(
        row: (&'static str, &'static str, &'static str, &'static str, Option<&'static str>),
        w: Width,
    ) -> Option<&'static str> {
        match w {
            Width::W64 => Some(row.0),
            Width::W32 => Some(row.1),
            Width::W16 => Some(row.2),
            Width::W8L => Some(row.3),
            Width::W8H => row.4,
        }
    }

    let key = any.to_ascii_lowercase();
    for row in T {
        if [row.0, row.1, row.2, row.3].iter().any(|&n| n == key) || row.4 == Some(key.as_str()) {
            return pick(row, to);
        }
    }
    None
}
